"""Reusable restricted Stack v3 row-group acquisition with preserved whole-unit attempts."""

import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from speck.data.acquisition_units import _bound_identity, _digest, _unit_config
from speck.data.production_rehearsal import _document_rejection, _gitleaks_filter, _record
from speck.data.stack_edu_stock import count_code_tokens, load_stack_edu_preparation
from speck.data.stack_v3_probe import content_rejection, metadata_rejection
from speck.provenance.io import durable_json, file_sha256


def load_stack_v3_acquisition(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if (
        value.get("format") != "speck_stack_v3_stock_acquisition"
        or value.get("format_version") != 1
        or value.get("source_id") != "stack_v3_train_permissive"
        or value.get("training_authority") is not False
        or value.get("checkpoint_rows") != 16
        or value.get("unit_boundary") != "one_complete_parquet_row_group"
        or value.get("repository_policy") != "natural_postfilter_no_tokenizer_sample_repository_cap"
        or value.get("maximum_acquisition_output_bytes") != 8589934592
        or value.get("minimum_free_bytes") != 68719476736
    ):
        raise ValueError("unsupported bounded Stack v3 acquisition contract")
    probe_id = _bound_identity(value["cached_probe_plan"], path.parent)
    probe = json.loads(Path(probe_id["path"]).read_text())
    census_id = _bound_identity(probe["census_plan"], Path(probe_id["path"]).parent)
    census = json.loads(Path(census_id["path"]).read_text())
    # Original census paths are repository-relative, including compatibility links.
    root = Path(census_id["path"]).parents[2]
    inputs = {
        key: _bound_identity(census[key], root)
        for key in ("source_qualification", "source_refinement", "source_use", "code_languages")
    }
    qualification, refinement, rights, languages = (
        json.loads(Path(inputs[key]["path"]).read_text())
        for key in ("source_qualification", "source_refinement", "source_use", "code_languages")
    )
    common_id = _bound_identity(probe["common_stock_plan"], Path(probe_id["path"]).parent)
    common = load_stack_edu_preparation(common_id["path"])
    raw_id = _bound_identity(value["targeted_raw_result"], path.parent)
    raw_result = json.loads(Path(raw_id["path"]).read_text())
    raw_plan_id = _bound_identity(raw_result["plan"], Path(raw_id["path"]).parent)
    raw_plan = json.loads(Path(raw_plan_id["path"]).read_text())
    if (
        rights["status"] != "all_sources_human_approved"
        or rights["automated_approval_made"] is not False
        or value["source_id"] not in rights["approved_source_ids"]
        or common["base"]["rights_record"]["sha256"] != inputs["source_use"]["sha256"]
        or _bound_identity(common["code_languages"], Path(common_id["path"]).parent)
        != inputs["code_languages"]
        or raw_result["status"] != "complete_raw_files_verified_not_content_qualified_stock"
        or any(
            raw_plan["source"][key] != qualification["source"][key] for key in ("repo", "revision")
        )
        or qualification["filters"]["exclude_vendor"] is not True
        or qualification["filters"]["exclude_forks"] is not True
        or qualification["filters"]["license_type"] != "permissive"
    ):
        raise ValueError("approved source, language, or completed raw identity differs")
    expected = [
        {
            "path": str(Path(census["raw_directory"]) / row["path"]),
            "filename": row["path"],
            "bytes": row["size"],
            "sha256": row["sha256"],
        }
        for row in qualification["source"]["files"]
    ]
    expected.extend(
        {
            "path": row["raw_path"],
            "filename": row["source_file"]["path"],
            "bytes": row["source_file"]["bytes"],
            "sha256": row["source_file"]["sha256"],
        }
        for row in raw_result["files"]
    )
    if len(expected) != 14 or len(value["files"]) != len(expected):
        raise ValueError("fourteen complete-file identities required")
    base = dict(common["base"])
    base.pop("stack_edu_policy")
    base["stack_v3_policy"] = {
        "languages": list(languages["language_weights_percent"]),
        "accepted_detected_licenses": refinement["license_policy"]["accepted_detected_licenses"],
        "English_prose": refinement["English_prose"],
        "excluded_path_components": census["excluded_path_components"],
        **{key: qualification["filters"][key] for key in ("min_file_bytes", "max_file_bytes")},
        "source_use": inputs["source_use"],
        "source_qualification": inputs["source_qualification"],
        "source_refinement": inputs["source_refinement"],
        "repository_policy": value["repository_policy"],
        "tokenizer": common["reference_tokenizer"],
    }
    units = []
    for index, (raw, prior) in enumerate(zip(value["files"], expected, strict=True)):
        if {key: raw[key] for key in prior} != prior:
            raise ValueError("complete raw source order or identity changed")
        offset = 0
        for group, count in enumerate(raw["row_group_rows"]):
            if type(count) is not int or count <= 0:
                raise ValueError("invalid complete row-group count")
            units.append(
                {
                    "id": f"stack_v3__file_{index:03d}__group_{group:03d}",
                    "category": "code",
                    "raw": raw,
                    "row_group": group,
                    "start_row": offset,
                    "stop_row": offset + count,
                    "source": {key: qualification["source"][key] for key in ("repo", "revision")},
                }
            )
            offset += count
    return {
        **value,
        "base": base,
        "units": units,
        "reference_tokenizer": common["reference_tokenizer"],
        "source_use": {**rights, "identity": inputs["source_use"]},
    }


def verify_raw(raw):
    path = Path(raw["path"])
    if path.stat().st_size != raw["bytes"] or file_sha256(path) != raw["sha256"]:
        raise ValueError("complete Stack v3 file changed")
    parquet = pq.ParquetFile(path)
    if [parquet.metadata.row_group(i).num_rows for i in range(parquet.num_row_groups)] != raw[
        "row_group_rows"
    ]:
        raise ValueError("complete raw row-group declaration differs")


def reopen_unit(directory, config):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("status") != "complete_not_training_data" or manifest[
        "config_sha256"
    ] != _digest(config):
        raise ValueError("Stack v3 unit owner changed")
    for entry in (manifest["output"], manifest["security_report"], manifest["unscanned_output"]):
        path = (directory / entry["path"]).resolve()
        if not path.is_relative_to(directory.resolve()) or file_sha256(path) != entry["sha256"]:
            raise ValueError("completed Stack v3 unit payload changed")
    return manifest


def acquire_stack_v3_unit(
    plan, unit, output, contamination, *, interrupt_after_repositories=None, remaining_bytes=None
):
    """Caller verifies full raw inputs once; each attempt processes one complete row group."""
    directory = Path(output) / unit["id"]
    config = _unit_config(plan, unit)
    directory.mkdir(parents=True, exist_ok=True)
    owner = directory / "config.json"
    if owner.exists():
        if json.loads(owner.read_text()) != config:
            raise ValueError("Stack v3 unit configuration changed")
    else:
        if list(directory.iterdir()):
            raise ValueError("unowned Stack v3 acquisition output")
        durable_json(owner, config)
    if (directory / "manifest.json").exists():
        return {"manifest": reopen_unit(directory, config), "reused": True}
    attempt = directory / f"attempt-{len(list(directory.glob('attempt-*'))):05d}"
    attempt.mkdir()
    started = time.perf_counter()
    durable_json(attempt / "execution.json", {"config_sha256": _digest(config)})
    unscanned = attempt / "unscanned.jsonl"
    policy = plan["base"]["stack_v3_policy"]
    reasons, metadata_counts, metadata_bytes = Counter(), Counter(), Counter()
    repositories = physical = retained = 0
    try:
        with unscanned.open("xb") as handle:
            try:
                parquet = pq.ParquetFile(unit["raw"]["path"])
                for batch in parquet.iter_batches(
                    row_groups=[unit["row_group"]],
                    columns=[
                        "repo_path",
                        "repo_id",
                        "commit_id",
                        "github_metadata.is_fork",
                        "files",
                    ],
                    batch_size=plan["checkpoint_rows"],
                    use_threads=False,
                ):
                    for repo in batch.to_pylist():
                        row = unit["start_row"] + repositories
                        for file_index, file in enumerate(repo.get("files") or []):
                            physical += 1
                            reason = metadata_rejection(repo, file, policy)
                            if reason is None:
                                metadata_counts[file["language"]] += 1
                                metadata_bytes[file["language"]] += file["size_bytes"]
                                reason = content_rejection(file, policy)
                            text = file.get("content")
                            if (
                                reason is None
                                and not plan["base"]["filtering"]["min_chars"]
                                <= len(text)
                                <= plan["base"]["filtering"]["max_chars"]
                            ):
                                reason = "code_character_envelope"
                            if reason is None:
                                reason = _document_rejection(
                                    {"content": text, "metadata": {}},
                                    plan["base"]["security"],
                                    contamination,
                                )
                            if reason:
                                reasons[reason] += 1
                                continue
                            metadata = {key: val for key, val in file.items() if key != "content"}
                            record = {
                                **_record(text, {"blob": file["content_id"]}),
                                "metadata": metadata,
                                "language": file["language"],
                                "repo_path": repo["repo_path"],
                                "repo_id": repo.get("repo_id"),
                                "commit_id": repo["commit_id"],
                                "file_path": file["file_path"],
                                "source_repo": unit["source"]["repo"],
                                "source_revision": unit["source"]["revision"],
                                "source_file": unit["raw"]["filename"],
                                "source_file_sha256": unit["raw"]["sha256"],
                                "source_row": row,
                                "source_file_index": file_index,
                                "released_repository_is_fork": (
                                    repo.get("github_metadata") or {}
                                ).get("is_fork"),
                            }
                            payload = (
                                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                            ).encode()
                            # Unscanned + scan input + possible security-clean rewrite coexist.
                            if (
                                remaining_bytes is not None
                                and 3 * (handle.tell() + len(payload)) > remaining_bytes
                            ):
                                raise ValueError("bounded acquisition output allowance exhausted")
                            handle.write(payload)
                            retained += 1
                        repositories += 1
                        if repositories == interrupt_after_repositories:
                            raise RuntimeError("injected row-group interruption")
            finally:
                handle.flush()
                os.fsync(handle.fileno())
        if repositories != unit["stop_row"] - unit["start_row"] or physical != retained + sum(
            reasons.values()
        ):
            raise ValueError("row-group outcome accounting does not conserve physical input")
        final = attempt / "records.jsonl"
        shutil.copyfile(unscanned, final)
        removed, security = _gitleaks_filter(
            final, plan["base"]["security"]["gitleaks_binary"]["path"], attempt / "security"
        )
        with final.open("rb") as handle:
            os.fsync(handle.fileno())
        counts = count_code_tokens(final, plan["reference_tokenizer"])
        if counts["documents"] != retained - removed:
            raise ValueError("security accounting differs from final record count")
        with final.open() as handle:
            utf8 = sum(len(json.loads(line)["text"].encode()) for line in handle)

        def identity(p):
            return {
                "path": str(p.relative_to(directory)),
                "sha256": file_sha256(p),
                "bytes": p.stat().st_size,
            }

        manifest = {
            "format": "speck_acquisition_unit",
            "format_version": 1,
            "status": "complete_not_training_data",
            "config_sha256": _digest(config),
            "unit_id": unit["id"],
            "category": "code",
            "row_window": [unit["start_row"], unit["stop_row"]],
            "repositories_seen": repositories,
            "physical_files_seen": physical,
            "yielded_rows": physical,
            "metadata_eligible_files_by_language": dict(metadata_counts),
            "metadata_eligible_bytes_by_language": dict(metadata_bytes),
            "rejections": {**reasons, "gitleaks": removed},
            "retained_records": counts["documents"],
            "retained_utf8_bytes": utf8,
            "tokens_before_full_exclusion": counts["tokens"],
            "language_counts_before_full_exclusion": counts["by_language"],
            "output": identity(final),
            "unscanned_output": identity(unscanned),
            "security_report": {
                **security,
                "path": str(Path(security["path"]).relative_to(directory)),
            },
            "training_authority": False,
        }
        # Publish only after all scan, count and output identities are available.
        durable_json(directory / "manifest.json", manifest)
        durable_json(
            attempt / "result.json",
            {"status": "complete", "elapsed_seconds": time.perf_counter() - started},
        )
        return {"manifest": reopen_unit(directory, config), "reused": False}
    except BaseException as error:
        durable_json(
            attempt / "result.json",
            {
                "status": "failed_preserved",
                "error_type": type(error).__name__,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        raise
