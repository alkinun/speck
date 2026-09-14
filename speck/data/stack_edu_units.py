"""Acquire deterministic Stack-Edu language units with durable row/token checkpoints."""

import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

from speck.data.acquisition_units import _digest, _unit_config
from speck.data.production_rehearsal import (
    _contamination_indexes,
    _document_rejection,
    _gitleaks_filter,
)
from speck.data.stack_edu_metadata import REQUIRED_COLUMNS, verify_metadata_file
from speck.data.stack_edu_stock import decode_code, metadata_rejection
from speck.data.swh_cache import fetch_cached_blob
from speck.provenance.io import durable_json, file_sha256
from speck.tokenization.tokenizer import Tokenizer


def _batches(path, start, stop, size):
    parquet = pq.ParquetFile(path)
    offset = 0
    for group in range(parquet.num_row_groups):
        end = offset + parquet.metadata.row_group(group).num_rows
        if end <= start:
            offset = end
            continue
        for batch in parquet.iter_batches(
            row_groups=[group], columns=sorted(REQUIRED_COLUMNS), batch_size=size, use_threads=False
        ):
            low, high = max(0, start - offset), min(batch.num_rows, stop - offset)
            if high > low:
                yield [
                    (offset + low + i, row)
                    for i, row in enumerate(batch.slice(low, high - low).to_pylist())
                ]
            offset += batch.num_rows
            if offset >= stop:
                return


def _reopen(manifest, directory, config):
    if (
        manifest.get("config_sha256") != _digest(config)
        or manifest.get("status") != "complete_not_training_data"
    ):
        raise ValueError("Stack-Edu acquisition manifest owner mismatch")
    for entry in (manifest["output"], manifest["security_report"], *manifest["fetch_batches"]):
        path = (directory / entry["path"]).resolve()
        if not path.is_relative_to(directory.resolve()) or file_sha256(path) != entry["sha256"]:
            raise ValueError("Stack-Edu acquisition payload changed")


def acquire_stack_edu_unit(
    plan, unit, output, contamination, tokenizer, executor, *, interrupt_after_rows=None
):
    started = time.perf_counter()
    directory = Path(output) / unit["id"]
    config = _unit_config(plan, unit)
    if directory.exists():
        if json.loads((directory / "config.json").read_text()) != config:
            raise ValueError("Stack-Edu acquisition owner/config differs")
    else:
        directory.mkdir(parents=True)
        durable_json(directory / "config.json", config)
    verify_metadata_file(unit["metadata_path"], unit)
    raw_metadata = {
        "path": unit["metadata_path"],
        **{key: unit["raw"][key] for key in ("sha256", "bytes")},
    }
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        _reopen(manifest, directory, config)
        return {
            "manifest": manifest,
            "raw": raw_metadata,
            "reused": True,
            "elapsed_seconds": time.perf_counter() - started,
        }
    state_path, scratch = directory / "state.json", directory / "unscanned.jsonl"
    if state_path.exists():
        checkpoint = json.loads(state_path.read_text())
        if checkpoint["config_sha256"] != _digest(config) or checkpoint["state_sha256"] != _digest(
            checkpoint["state"]
        ):
            raise ValueError("Stack-Edu checkpoint identity mismatch")
        state = checkpoint["state"]
    else:
        state = {
            "next_row": unit["start_row"],
            "seen_rows": 0,
            "accepted_records": 0,
            "candidate_tokens": 0,
            "retained_utf8_bytes": 0,
            "rejections": {},
            "output_bytes": 0,
            "output_sha256": hashlib.sha256(b"").hexdigest(),
        }
    if not unit["start_row"] <= state["next_row"] <= unit["stop_row"]:
        raise ValueError("Stack-Edu checkpoint cursor outside declared metadata window")
    policy = plan["base"]["stack_edu_policy"]
    hasher = hashlib.sha256()
    with scratch.open("r+b" if scratch.exists() else "w+b") as handle:
        remaining = state["output_bytes"]
        while remaining:
            chunk = handle.read(min(remaining, 8 * 1024 * 1024))
            if not chunk:
                raise ValueError("Stack-Edu checkpoint output is truncated")
            hasher.update(chunk)
            remaining -= len(chunk)
        if hasher.hexdigest() != state["output_sha256"]:
            raise ValueError("Stack-Edu checkpoint output checksum mismatch")
        if scratch.stat().st_size > state["output_bytes"]:
            tail = directory / f"uncommitted-tail-{time.time_ns()}.jsonl"
            with tail.open("xb") as dest:
                shutil.copyfileobj(handle, dest, length=8 * 1024 * 1024)
                dest.flush()
                os.fsync(dest.fileno())
            durable_json(
                tail.with_suffix(".receipt.json"),
                {
                    "path": tail.name,
                    "sha256": file_sha256(tail),
                    "bytes": tail.stat().st_size,
                    "checkpoint": checkpoint if state_path.exists() else None,
                },
            )
        handle.truncate(state["output_bytes"])
        handle.seek(state["output_bytes"])

        def checkpoint_state():
            handle.flush()
            os.fsync(handle.fileno())
            state["output_bytes"] = handle.tell()
            state["output_sha256"] = hasher.hexdigest()
            durable_json(
                state_path,
                {"config_sha256": _digest(config), "state": state, "state_sha256": _digest(state)},
            )

        def reject(reason):
            state["rejections"][reason] = state["rejections"].get(reason, 0) + 1

        if state["candidate_tokens"] < unit["candidate_target_tokens"]:
            for rows in _batches(
                unit["metadata_path"], state["next_row"], unit["stop_row"], plan["checkpoint_rows"]
            ):
                reasons = [metadata_rejection(row, unit["language"], policy) for _, row in rows]
                blob_ids = list(
                    dict.fromkeys(
                        row["blob_id"]
                        for (_, row), reason in zip(rows, reasons, strict=True)
                        if reason is None
                    )
                )
                fetched = dict(
                    zip(
                        blob_ids,
                        executor.map(
                            lambda blob: fetch_cached_blob(
                                blob, plan["blob_cache"], policy["fetch"]
                            ),
                            blob_ids,
                        ),
                        strict=True,
                    )
                )
                # Preserve every prefetched input, including any beyond the eventual quota stop.
                batch_path = (
                    directory
                    / "fetch-batches"
                    / f"rows-{rows[0][0]:010d}-{rows[-1][0] + 1:010d}.json"
                )
                batch_receipt = {
                    "metadata_sha256": unit["raw"]["sha256"],
                    "rows": [rows[0][0], rows[-1][0] + 1],
                    "blobs": [
                        {"blob_id": blob, "manifest": receipt}
                        for blob, (_, receipt) in fetched.items()
                    ],
                }
                if batch_path.exists() and json.loads(batch_path.read_text()) != batch_receipt:
                    raise ValueError("Stack-Edu replay fetched-input identities changed")
                durable_json(batch_path, batch_receipt)
                for (index, row), reason in zip(rows, reasons, strict=True):
                    if state["candidate_tokens"] >= unit["candidate_target_tokens"]:
                        break
                    state["next_row"] = index + 1
                    state["seen_rows"] += 1
                    if reason:
                        reject(reason)
                        continue
                    raw, receipt = fetched[row["blob_id"]]
                    if raw is None:
                        reject("blob_missing_404")
                        continue
                    reason, text, prose = decode_code(row, raw, policy)
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
                        reject(reason)
                        continue
                    count = len(tokenizer.encode(text, bos=True, eos=True))
                    record = {
                        "text": text,
                        "source": "stack_edu",
                        "content_id": row["blob_id"],
                        "released_content_sha256": hashlib.sha256(raw).hexdigest(),
                        "source_repo": unit["reader"]["repo"],
                        "source_revision": unit["reader"]["revision"],
                        "source_file": unit["raw"]["filename"],
                        "source_row": index,
                        "repo_path": row["repo_name"],
                        "file_path": row["path"],
                        "language": unit["language"],
                        "detected_licenses": row["detected_licenses"],
                        "license_type": row["license_type"],
                        "score": row["score"],
                        "int_score": row["int_score"],
                        "src_encoding": row["src_encoding"],
                        "declared_length_bytes": row["length_bytes"],
                        "metadata_file_sha256": unit["raw"]["sha256"],
                        "swh_blob": receipt,
                        "url": None,
                        "host": None,
                        "commit_id": None,
                        "vendor_metadata": "not_released_path_rule_only",
                        **prose,
                    }
                    encoded = (
                        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                    ).encode()
                    handle.write(encoded)
                    hasher.update(encoded)
                    state["accepted_records"] += 1
                    state["candidate_tokens"] += count
                    state["retained_utf8_bytes"] += len(raw)
                checkpoint_state()
                if interrupt_after_rows is not None and state["seen_rows"] >= interrupt_after_rows:
                    raise RuntimeError("injected Stack-Edu acquisition interruption")
                if state["seen_rows"] % (plan["checkpoint_rows"] * 32) == 0:
                    print(
                        f"{unit['id']}: row {state['next_row']} / candidate tokens {state['candidate_tokens']}",
                        flush=True,
                    )
                if state["candidate_tokens"] >= unit["candidate_target_tokens"]:
                    break
        checkpoint_state()
    final = directory / "records.jsonl"
    # Rebuild only the owned unpublished security-filter input; its immutable
    # checkpoint source and any earlier scanner reports remain preserved.
    if final.exists():
        final.rename(directory / f"unpublished-security-input-{time.time_ns()}.jsonl")
    shutil.copyfile(scratch, final)
    security_dir = directory / f"security-{time.time_ns()}"
    removed, security = _gitleaks_filter(
        final, plan["base"]["security"]["gitleaks_binary"]["path"], security_dir
    )
    records = utf8_bytes = 0
    with final.open() as handle:
        for raw in handle:
            records += 1
            utf8_bytes += len(json.loads(raw)["text"].encode())
    batches = [
        {"path": str(p.relative_to(directory)), "sha256": file_sha256(p)}
        for p in sorted((directory / "fetch-batches").glob("*.json"))
    ]
    manifest = {
        "format": "speck_acquisition_unit",
        "format_version": 1,
        "status": "complete_not_training_data",
        "config_sha256": _digest(config),
        "unit_id": unit["id"],
        "category": "code",
        "language": unit["language"],
        "row_window": [unit["start_row"], state["next_row"]],
        "declared_stop_row": unit["stop_row"],
        "yielded_rows": state["seen_rows"],
        "rejections": {**state["rejections"], "gitleaks": removed},
        "retained_records": records,
        "retained_utf8_bytes": utf8_bytes,
        "pre_gitleaks_candidate_tokens": state["candidate_tokens"],
        "candidate_target_tokens": unit["candidate_target_tokens"],
        "candidate_target_pass": state["candidate_tokens"] >= unit["candidate_target_tokens"],
        "output": {"path": final.name, "sha256": file_sha256(final), "bytes": final.stat().st_size},
        "security_report": {**security, "path": str(Path(security["path"]).relative_to(directory))},
        "fetch_batches": batches,
        "training_authority": False,
    }
    for durable_path in (final, Path(security["path"])):
        with durable_path.open("rb") as handle:
            os.fsync(handle.fileno())
    durable_json(manifest_path, manifest)
    return {
        "manifest": manifest,
        "raw": raw_metadata,
        "elapsed_seconds": time.perf_counter() - started,
        "reused": False,
    }


def prepare_stack_edu_units(plan, output):
    started = time.perf_counter()
    contamination = _contamination_indexes(plan["base"])
    tokenizer_id = plan["base"]["stack_edu_policy"]["tokenizer"]
    if file_sha256(tokenizer_id["path"]) != tokenizer_id["sha256"]:
        raise ValueError("Stack-Edu acquisition tokenizer changed")
    tokenizer = Tokenizer(tokenizer_id["path"])
    setup = time.perf_counter() - started
    reports = []
    with ThreadPoolExecutor(
        max_workers=plan["base"]["stack_edu_policy"]["fetch"]["workers"]
    ) as executor:
        for unit in plan["units"]:
            print(f"acquiring Stack-Edu {unit['language']}", flush=True)
            attempt_path = Path(output) / unit["id"] / f"attempt-{time.time_ns()}.json"
            unit_started = time.perf_counter()
            try:
                result = acquire_stack_edu_unit(
                    plan, unit, output, contamination, tokenizer, executor
                )
            except BaseException as error:
                durable_json(
                    attempt_path,
                    {
                        "status": "failed_or_interrupted",
                        "error_type": type(error).__name__,
                        "elapsed_seconds": time.perf_counter() - unit_started,
                    },
                )
                raise
            durable_json(attempt_path, result)
            reports.append(result)
    return {
        "units": reports,
        "setup_seconds": setup,
        "elapsed_seconds": time.perf_counter() - started,
    }
