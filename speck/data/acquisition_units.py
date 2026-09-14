"""Bounded, independently resumable raw-source units and ordered cohort deduplication."""

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import speck.data.production_data as production_data
from speck.data.acquisition import _dataset_url, _download_file, iter_source_file_documents
from speck.data.configuration import _validate_source
from speck.data.production_rehearsal import (
    _contamination_indexes,
    _document_rejection,
    _gitleaks_filter,
    _raw_local_path,
    _record,
)
from speck.provenance.io import durable_json, file_sha256
from speck.provenance.repository import repository_artifact, repository_root


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _bound_identity(identity, directory):
    root = repository_root(__file__)
    path = (directory / identity["path"]).resolve()
    if not path.exists() and path.is_relative_to(root):
        path = repository_artifact(path.relative_to(root), root)
    if not path.is_file() or file_sha256(path) != identity["sha256"]:
        raise ValueError(f"acquisition unit input identity mismatch: {path}")
    return {"path": str(path), "sha256": identity["sha256"]}


def load_unit_plan(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if (
        set(value)
        != {
            "format",
            "format_version",
            "scope",
            "base_plan",
            "raw_files",
            "row_windows",
            "checkpoint_rows",
            "dedup_checkpoint_records",
            "raw_directory",
        }
        or value["format"] != "speck_acquisition_unit_rehearsal"
        or value["format_version"] != 1
        or value["scope"] != "bounded_raw_acquisition_and_cohort_dedup_not_training_data"
    ):
        raise ValueError("unsupported acquisition unit plan")
    base_identity = _bound_identity(value["base_plan"], path.parent)
    base_path = Path(base_identity["path"])
    base = json.loads(base_path.read_text())
    for name in ("source_registry", "rights_record", "deny_ledger", "contamination_plan"):
        base[name] = _bound_identity(base[name], base_path.parent)
    base["security"]["gitleaks_binary"] = _bound_identity(
        base["security"]["gitleaks_binary"], base_path.parent
    )
    rights = json.loads(Path(base["rights_record"]["path"]).read_text())
    if (
        rights.get("status") != "all_sources_human_approved"
        or rights.get("automated_approval_made") is not False
    ):
        raise ValueError("acquisition units require the existing source approval")
    sources = {item["id"]: item for item in base["sources"]}
    if {item["source_id"] for item in value["raw_files"]} != set(sources) or len(
        value["raw_files"]
    ) != len(sources):
        raise ValueError("raw files must cover exactly the six base-plan sources")
    if not set(sources) <= set(rights["approved_source_ids"]):
        raise ValueError("unapproved acquisition source")
    windows = value["row_windows"]
    if not isinstance(windows, list) or not windows:
        raise ValueError("acquisition requires row windows")
    end = 0
    for window in windows:
        if (
            not isinstance(window, list)
            or len(window) != 2
            or any(isinstance(n, bool) or not isinstance(n, int) for n in window)
            or not end <= window[0] < window[1] <= 4096
        ):
            raise ValueError(
                "acquisition windows must be ordered, nonoverlapping, and within 4096 rows"
            )
        end = window[1]
    for name in ("checkpoint_rows", "dedup_checkpoint_records"):
        if (
            isinstance(value[name], bool)
            or not isinstance(value[name], int)
            or not 1 <= value[name] <= 10000
        ):
            raise ValueError("invalid acquisition checkpoint interval")
    units = []
    for raw in value["raw_files"]:
        if set(raw) != {"source_id", "filename", "sha256", "bytes"}:
            raise ValueError("invalid raw file identity")
        if (
            Path(raw["filename"]).is_absolute()
            or ".." in Path(raw["filename"]).parts
            or not isinstance(raw["sha256"], str)
            or len(raw["sha256"]) != 64
            or isinstance(raw["bytes"], bool)
            or not isinstance(raw["bytes"], int)
            or not 0 < raw["bytes"] <= 3_000_000_000
        ):
            raise ValueError("raw file exceeds the bounded identity contract")
        source = sources[raw["source_id"]]
        reader = _validate_source(source["reader"])
        for start, stop in windows:
            units.append(
                {
                    "id": f"{source['id']}__rows_{start}_{stop}",
                    "category": source["category"],
                    "reader": reader,
                    "raw": raw,
                    "start_row": start,
                    "stop_row": stop,
                }
            )
    raw_directory = (path.parent / value["raw_directory"]).resolve()
    return {
        **value,
        "base": base,
        "base_plan": base_identity,
        "units": units,
        "raw_directory": str(raw_directory),
    }


def _raw_file(plan, unit):
    reader, raw = unit["reader"], unit["raw"]
    path = _raw_local_path(Path(plan["raw_directory"]), reader, reader["revision"], raw["filename"])
    downloaded = not path.exists()
    started = time.perf_counter()
    if downloaded:
        _download_file(
            _dataset_url(reader["repo"], reader["revision"], raw["filename"]),
            path,
            unit["id"],
            repo=reader["repo"],
        )
    download_seconds = time.perf_counter() - started
    started = time.perf_counter()
    if path.stat().st_size != raw["bytes"] or file_sha256(path) != raw["sha256"]:
        raise ValueError(f"raw acquisition file identity mismatch: {path}")
    return {
        "path": str(path),
        "sha256": raw["sha256"],
        "bytes": raw["bytes"],
        "downloaded_this_invocation": downloaded,
        "download_seconds": download_seconds,
        "verification_seconds": time.perf_counter() - started,
    }


def _unit_config(plan, unit):
    if "math_english" in plan["base"] and unit["category"] != "math":
        raise ValueError("math-prose language policy may only govern math units")
    if "science_filters" in plan["base"] and unit["category"] != "science":
        raise ValueError("science language/license policy may only govern science units")
    if "finemath_filters" in plan["base"] and unit["category"] != "math":
        raise ValueError("FineMath metadata policy may only govern math units")
    if "cosmopedia_policy" in plan["base"] and unit["category"] != "synthetic":
        raise ValueError("Cosmopedia lineage policy may only govern synthetic units")
    if "stack_edu_policy" in plan["base"] and unit["category"] != "code":
        raise ValueError("Stack-Edu policy may only govern code units")
    if "fineweb_edu_filters" in plan["base"] and unit["category"] != "web":
        raise ValueError("FineWeb-Edu policy may only govern web units")
    config = {
        "unit": unit,
        "filtering": plan["base"]["filtering"],
        "security": plan["base"]["security"],
        "contamination_plan": plan["base"]["contamination_plan"],
        "checkpoint_rows": plan["checkpoint_rows"],
    }
    for key in (
        "math_english",
        "source_use_extension",
        "science_filters",
        "finemath_filters",
        "fineweb_edu_filters",
        "cosmopedia_policy",
        "stack_edu_policy",
    ):
        if key in plan["base"]:
            config[key] = plan["base"][key]
    return config


def acquire_unit(plan, unit, output_root, contamination, *, interrupt_after_rows=None):
    """Resume one physical row window; no tokenizer participates in acquisition."""

    started = time.perf_counter()
    config = _unit_config(plan, unit)
    directory = Path(output_root) / unit["id"]
    owner = directory / "config.json"
    if directory.exists():
        if not owner.is_file() or json.loads(owner.read_text()) != config:
            raise ValueError("acquisition unit output has a different owner/config")
    else:
        directory.mkdir(parents=True)
        durable_json(owner, config)
    raw_identity = _raw_file(plan, unit)
    if "expected_file_rows" in unit:
        import io

        import pyarrow as pa
        import pyarrow.parquet as pq

        if unit["reader"]["file_format"] == "parquet":
            rows = pq.ParquetFile(raw_identity["path"]).metadata.num_rows
        elif unit["reader"]["file_format"] == "jsonl_zstd":
            with io.TextIOWrapper(
                pa.input_stream(raw_identity["path"], compression="zstd"), encoding="utf-8"
            ) as handle:
                rows = sum(1 for _ in handle)
        else:
            raise ValueError("complete row-count verification is unavailable for this format")
        if rows != unit["expected_file_rows"]:
            raise ValueError("complete acquisition shard row count differs from its plan")
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("config_sha256") != _digest(config):
            raise ValueError("acquisition unit manifest config mismatch")
        for item in (manifest["output"], manifest["security_report"]):
            if file_sha256(directory / item["path"]) != item["sha256"]:
                raise ValueError("acquisition unit published output identity mismatch")
        return {
            "manifest": manifest,
            "raw": raw_identity,
            "elapsed_seconds": time.perf_counter() - started,
            "reused": True,
        }
    state_path = directory / "state.json"
    scratch = directory / "unscanned.jsonl"
    checkpoint_record = json.loads(state_path.read_text()) if state_path.exists() else None
    if checkpoint_record is not None and (
        checkpoint_record.get("config_sha256") != _digest(config)
        or checkpoint_record.get("state_sha256") != _digest(checkpoint_record.get("state"))
    ):
        raise ValueError("acquisition checkpoint state identity mismatch")
    state = (
        checkpoint_record["state"]
        if checkpoint_record is not None
        else {
            "next_row": unit["start_row"],
            "yielded_rows": 0,
            "accepted_rows": 0,
            "rejections": {},
            "output_bytes": 0,
            "output_sha256": hashlib.sha256(b"").hexdigest(),
        }
    )
    if not unit["start_row"] <= state["next_row"] <= unit["stop_row"]:
        raise ValueError("acquisition unit checkpoint row is outside its window")
    hasher = hashlib.sha256()
    with scratch.open("r+b" if scratch.exists() else "w+b") as handle:
        remaining = state["output_bytes"]
        while remaining:
            chunk = handle.read(min(remaining, 8 * 1024 * 1024))
            if not chunk:
                raise ValueError("acquisition checkpoint text is truncated")
            hasher.update(chunk)
            remaining -= len(chunk)
        if hasher.hexdigest() != state["output_sha256"]:
            raise ValueError("acquisition checkpoint text hash mismatch")
        handle.truncate(state["output_bytes"])
        handle.seek(state["output_bytes"])

        def checkpoint():
            handle.flush()
            os.fsync(handle.fileno())
            state["output_bytes"] = handle.tell()
            state["output_sha256"] = hasher.hexdigest()
            durable_json(
                state_path,
                {"config_sha256": _digest(config), "state": state, "state_sha256": _digest(state)},
            )

        if state["next_row"] < unit["stop_row"]:
            documents = iter_source_file_documents(
                source=unit["reader"],
                revision=unit["reader"]["revision"],
                filename=unit["raw"]["filename"],
                filtering=plan["base"]["filtering"],
                cache_dir=plan["raw_directory"],
                keep_raw=True,
                start_row=state["next_row"],
                stop_row=unit["stop_row"],
            )
            try:
                for document in documents:
                    state["next_row"] = document["row"] + 1
                    state["yielded_rows"] += 1
                    reason = _document_rejection(document, plan["base"]["security"], contamination)
                    english_probability = None
                    synthetic_metadata = None
                    if reason is None and "cosmopedia_policy" in plan["base"]:
                        from speck.data.cosmopedia_stock import cosmopedia_document

                        reason, synthetic_metadata = cosmopedia_document(
                            document, plan["base"]["cosmopedia_policy"]
                        )
                    if reason is None and "fineweb_edu_filters" in plan["base"]:
                        from speck.data.fineweb_edu_stock import fineweb_edu_rejection

                        reason, english_probability = fineweb_edu_rejection(
                            document, plan["base"]["fineweb_edu_filters"]
                        )
                    if reason is None and "finemath_filters" in plan["base"]:
                        from speck.data.finemath_stock import finemath_rejection

                        reason = finemath_rejection(document, plan["base"]["finemath_filters"])
                    if reason is None and "science_filters" in plan["base"]:
                        from speck.data.science_stock import science_rejection

                        reason, english_probability = science_rejection(
                            document, plan["base"]["science_filters"]
                        )
                    if reason is None and "math_english" in plan["base"]:
                        from speck.data.sources.math_sample import _language_result

                        language, english_probability = _language_result(
                            document["content"], None, plan["base"]["math_english"]
                        )
                        if language != "English":
                            reason = f"math_prose_{language}"
                    if reason:
                        state["rejections"][reason] = state["rejections"].get(reason, 0) + 1
                    else:
                        metadata = (
                            synthetic_metadata
                            if synthetic_metadata is not None
                            else document.get("metadata") or {}
                        )
                        record = {
                            **_record(document["content"], metadata),
                            "metadata": metadata,
                            "score": document.get("score"),
                            "source_repo": unit["reader"]["repo"],
                            "source_revision": unit["reader"]["revision"],
                            "source_file": document["file"],
                            "source_row": document["row"],
                        }
                        if synthetic_metadata is not None:
                            # This release has no document ID. Preserve a stable raw-file/row
                            # locator without treating the prompt or seed label as an ID.
                            record["content_id"] = (
                                f"{unit['reader']['id']}:{unit['raw']['sha256']}:{document['row']}"
                            )
                        if english_probability is not None:
                            record["detected_English_probability"] = english_probability
                        raw = (
                            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                        ).encode()
                        handle.write(raw)
                        hasher.update(raw)
                        state["accepted_rows"] += 1
                    if state["yielded_rows"] % plan["checkpoint_rows"] == 0:
                        checkpoint()
                    if state["yielded_rows"] == interrupt_after_rows:
                        raise RuntimeError("injected acquisition unit interruption")
            finally:
                documents.close()
        state["next_row"] = unit["stop_row"]
        checkpoint()
    final = directory / "records.jsonl"
    shutil.copyfile(scratch, final)
    removed, security = _gitleaks_filter(
        final, plan["base"]["security"]["gitleaks_binary"]["path"], directory / "security"
    )
    utf8_bytes = records = 0
    with final.open() as handle:
        for raw in handle:
            records += 1
            utf8_bytes += len(json.loads(raw)["text"].encode())
    manifest = {
        "format": "speck_acquisition_unit",
        "format_version": 1,
        "status": "complete_not_training_data",
        "config_sha256": _digest(config),
        "unit_id": unit["id"],
        "category": unit["category"],
        "row_window": [unit["start_row"], unit["stop_row"]],
        "yielded_rows": state["yielded_rows"],
        "rejections": {**state["rejections"], "gitleaks": removed},
        "retained_records": records,
        "retained_utf8_bytes": utf8_bytes,
        "output": {"path": final.name, "sha256": file_sha256(final), "bytes": final.stat().st_size},
        "security_report": {**security, "path": str(Path(security["path"]).relative_to(directory))},
        "training_authority": False,
    }
    durable_json(manifest_path, manifest)
    return {
        "manifest": manifest,
        "raw": raw_identity,
        "elapsed_seconds": time.perf_counter() - started,
        "reused": False,
    }


def prepare_units(plan, output, *, interrupt_unit=None, interrupt_after_rows=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    contamination = _contamination_indexes(plan["base"])
    setup_seconds = time.perf_counter() - started
    reports = []
    for unit in plan["units"]:
        report = acquire_unit(
            plan,
            unit,
            output,
            contamination,
            interrupt_after_rows=(interrupt_after_rows if unit["id"] == interrupt_unit else None),
        )
        reports.append(report)
        # Preserve each finished unit's cost even if the surrounding process is interrupted.
        attempt_path = output / unit["id"] / f"attempt-{time.time_ns()}.json"
        durable_json(attempt_path, report)
    return {
        "units": reports,
        "setup_seconds": setup_seconds,
        "elapsed_seconds": time.perf_counter() - started,
    }


def deduplicate_units(plan, acquired, output, *, crash_after_records=None):
    """Deduplicate the entire bounded cohort with a lower-precedence exact-replay control."""

    sources = []
    for unit in plan["units"]:
        directory = Path(acquired) / unit["id"]
        manifest = json.loads((directory / "manifest.json").read_text())
        if (
            manifest.get("format") != "speck_acquisition_unit"
            or manifest.get("status") != "complete_not_training_data"
            or manifest.get("config_sha256") != _digest(_unit_config(plan, unit))
            or manifest.get("output", {}).get("path") != "records.jsonl"
        ):
            raise ValueError("dedup input is not a completed unit of this plan")
        sources.append(
            {
                "id": unit["id"],
                "precedence": len(sources) + 1,
                "path": str(directory / manifest["output"]["path"]),
                "sha256": manifest["output"]["sha256"],
                "text_field": "text",
                "content_sha256_field": "released_content_sha256",
                "url_field": "url",
                "domain_field": "host",
                "blob_field": "content_id",
            }
        )
    sources.append({**sources[0], "id": "exact_replay_control", "precedence": len(sources) + 1})
    config = {
        "format": "speck_production_text_preprocess",
        "format_version": 1,
        "status": "fixture_or_rehearsal_authorized_not_training_authority",
        "sources": sources,
        "deny_ledger": plan["base"]["deny_ledger"],
        "policy": {
            k: v for k, v in plan["base"]["deduplication"].items() if k != "checkpoint_records"
        },
        "checkpoint_records": plan["dedup_checkpoint_records"],
        "cleanup_files": [],
        "output_directory": str(output),
    }
    original = production_data._signature

    def batched_signature(shingles, num_perm, seed):
        from datasketch import MinHash

        signature = MinHash(num_perm=num_perm, seed=seed)
        signature.update_batch(shingles)
        return signature

    started = time.perf_counter()
    production_data._signature = batched_signature
    try:
        result = production_data.preprocess_sources(config, crash_after_records=crash_after_records)
    finally:
        production_data._signature = original
    return {
        "result": result,
        "elapsed_seconds": time.perf_counter() - started,
        "firewall_reference_exclusion": "not_run_on_this_engineering_cohort",
        "training_authority": False,
    }
