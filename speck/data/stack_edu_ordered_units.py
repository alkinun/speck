"""Publish ordered Stack-Edu content batches and verified archival copies."""

import hashlib
import json
import os
import shutil
import tarfile
import time
from collections import Counter
from pathlib import Path

from speck.data.acquisition_units import _digest, _unit_config
from speck.data.ordered_blob_fetch import ordered_prefetch
from speck.data.production_rehearsal import _document_rejection, _gitleaks_filter
from speck.data.stack_edu_stock import count_code_tokens, decode_code
from speck.data.swh_cache import read_cached_blob
from speck.provenance.io import durable_json, file_sha256


def reopen_batch(directory, config):
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest["config_sha256"] != _digest(config)
        or manifest["status"] != "complete_not_training_data"
    ):
        raise ValueError("completed ordered unit owner changed")
    for key in ("output", "unscanned_output", "security_report", "fetch_journal"):
        item = manifest[key]
        path = (directory / item["path"]).resolve()
        if not path.is_relative_to(directory.resolve()) or file_sha256(path) != item["sha256"]:
            raise ValueError("completed ordered unit payload changed")
    return manifest


def acquire_batch(
    plan, unit, targets, output, store, contamination, tokenizer, *, interrupt_after=None
):
    directory = Path(output) / unit["id"]
    if not targets or unit["candidate_tokens_remaining"] <= 0:
        raise ValueError("ordered unit requires nonempty targets and positive remaining quota")
    if unit["targets_sha256"] != _digest(targets):
        raise ValueError("ordered unit target identities changed")
    config = _unit_config(plan, unit)
    directory.mkdir(parents=True, exist_ok=True)
    owner = directory / "config.json"
    if owner.exists():
        if json.loads(owner.read_text()) != config:
            raise ValueError("ordered content unit configuration changed")
    else:
        if list(directory.iterdir()):
            raise ValueError("unowned ordered content output")
        durable_json(owner, config)
    if (directory / "manifest.json").exists():
        return reopen_batch(directory, config)
    attempt = directory / f"attempt-{len(list(directory.glob('attempt-*'))):05d}"
    attempt.mkdir()
    started = time.perf_counter()
    durable_json(attempt / "execution.json", {"config_sha256": _digest(config)})
    journal = attempt / "fetches.jsonl"
    unscanned = attempt / "unscanned.jsonl"
    reasons = Counter()
    consumed = accepted = candidate_tokens = 0
    try:
        # Fetch the entire fixed batch. Lookahead remains preserved even when the
        # deterministic content-token prefix ends partway through this batch.
        with journal.open("xb") as handle:
            stream = ordered_prefetch(
                targets,
                store.fetch,
                workers=plan["workers"],
                window=plan["window"],
                key=lambda row: row["blob_id"],
            )
            try:
                for index, (target, receipt) in enumerate(stream):
                    row = {"index": index, "blob_id": target["blob_id"], "manifest": receipt}
                    handle.write((json.dumps(row, sort_keys=True) + "\n").encode())
                    if index + 1 == interrupt_after:
                        raise RuntimeError("injected ordered stock interruption")
            finally:
                stream.close()
                handle.flush()
                os.fsync(handle.fileno())
        policy = plan["base"]["stack_edu_policy"]
        with journal.open() as fetched, unscanned.open("xb") as handle:
            try:
                for target, line in zip(targets, fetched, strict=True):
                    if candidate_tokens >= unit["candidate_tokens_remaining"]:
                        break
                    receipt = json.loads(line)
                    if receipt["blob_id"] != target["blob_id"]:
                        raise ValueError("ordered content journal differs from target")
                    manifest_path = Path(receipt["manifest"]["path"])
                    if file_sha256(manifest_path) != receipt["manifest"]["sha256"]:
                        raise ValueError("fetched source manifest changed")
                    raw, _ = read_cached_blob(
                        manifest_path.parent, target["blob_id"], policy["fetch"]
                    )
                    consumed += 1
                    if raw is None:
                        reasons["blob_missing_404"] += 1
                        continue
                    metadata = target["metadata"]
                    reason, text, prose = decode_code(metadata, raw, policy)
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
                    count = len(tokenizer.encode(text, bos=True, eos=True))
                    record = {
                        "text": text,
                        "source": "stack_edu",
                        "content_id": target["blob_id"],
                        "released_content_sha256": hashlib.sha256(raw).hexdigest(),
                        "source_repo": unit["reader"]["repo"],
                        "source_revision": unit["reader"]["revision"],
                        "source_file": unit["raw"]["filename"],
                        "source_row": target["source_row"],
                        "eligible_ordinal": target["eligible_ordinal"],
                        "metadata_file_sha256": unit["raw"]["sha256"],
                        "repo_path": metadata["repo_name"],
                        "file_path": metadata["path"],
                        "language": unit["language"],
                        "metadata": metadata,
                        "swh_blob": receipt["manifest"],
                        "url": None,
                        "host": None,
                        "commit_id": None,
                        "vendor_metadata": "not_released_path_rule_only",
                        **prose,
                    }
                    handle.write(
                        (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
                    )
                    candidate_tokens += count
                    accepted += 1
            finally:
                handle.flush()
                os.fsync(handle.fileno())
        final = attempt / "records.jsonl"
        shutil.copyfile(unscanned, final)
        removed, security = _gitleaks_filter(
            final, plan["base"]["security"]["gitleaks_binary"]["path"], attempt / "security"
        )
        with final.open("rb") as handle:
            os.fsync(handle.fileno())
        counts = count_code_tokens(final, plan["reference_tokenizer"])
        if counts["documents"] + removed != accepted or consumed != accepted + sum(
            reasons.values()
        ):
            raise ValueError("ordered content outcome accounting failed")

        def identity(path):
            return {
                "path": str(path.relative_to(directory)),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }

        with final.open() as handle:
            utf8 = sum(len(json.loads(line)["text"].encode()) for line in handle)
        manifest = {
            "format": "speck_acquisition_unit",
            "format_version": 1,
            "status": "complete_not_training_data",
            "unit_id": unit["id"],
            "config_sha256": _digest(config),
            "category": "code",
            "language": unit["language"],
            "row_window": [targets[0]["source_row"], targets[consumed - 1]["source_row"] + 1],
            "eligible_window": [
                targets[0]["eligible_ordinal"],
                targets[0]["eligible_ordinal"] + consumed,
            ],
            "requested_eligible_rows": len(targets),
            "consumed_eligible_rows": consumed,
            "prefetched_rows_after_prefix": len(targets) - consumed,
            "rejections": {**reasons, "gitleaks": removed},
            "retained_records": counts["documents"],
            "retained_utf8_bytes": utf8,
            "pre_gitleaks_candidate_tokens": candidate_tokens,
            "tokens_before_full_exclusion": counts["tokens"],
            "output": identity(final),
            "unscanned_output": identity(unscanned),
            "fetch_journal": identity(journal),
            "security_report": {
                **security,
                "path": str(Path(security["path"]).relative_to(directory)),
            },
            "elapsed_seconds_this_attempt": time.perf_counter() - started,
            "training_authority": False,
        }
        durable_json(directory / "manifest.json", manifest)
        durable_json(attempt / "result.json", {"status": "complete"})
        return manifest
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


def archive_batch(directory, manifest, archive):
    """Archive the unit and every referenced blob, then reopen every archived payload."""
    directory, archive = Path(directory), Path(archive)
    archive.mkdir(parents=True, exist_ok=True)
    receipt_path = archive / "manifest.json"
    owner = file_sha256(directory / "manifest.json")
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if (
            receipt["unit_manifest_sha256"] != owner
            or file_sha256(receipt["tar"]["path"]) != receipt["tar"]["sha256"]
            or file_sha256(receipt["inventory"]["path"]) != receipt["inventory"]["sha256"]
        ):
            raise ValueError("completed archival unit changed")
        return receipt
    files = {
        "unit/" + str(path.relative_to(directory)): path
        for path in directory.rglob("*")
        if path.is_file()
    }
    # The completed journal covers the entire batch, including every target of any
    # earlier interrupted attempt. Earlier partial journals stay archived verbatim.
    with (directory / manifest["fetch_journal"]["path"]).open() as handle:
        for line in handle:
            row = json.loads(line)
            source = Path(row["manifest"]["path"])
            if file_sha256(source) != row["manifest"]["sha256"]:
                raise ValueError("archival blob input identity changed")
            for path in source.parent.iterdir():
                if path.is_file():
                    name = "blobs/" + row["blob_id"] + "/" + path.name
                    if name in files and file_sha256(files[name]) != file_sha256(path):
                        raise ValueError("conflicting archival blob identities")
                    files[name] = path
    attempt = archive / f"attempt-{len(list(archive.glob('attempt-*'))):05d}"
    attempt.mkdir()
    inventory = [
        {
            "path": name,
            "original_path": str(path),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for name, path in sorted(files.items())
    ]
    inventory_path = attempt / "inventory.json"
    durable_json(inventory_path, inventory)
    bundle = attempt / "unit.tar"
    with tarfile.open(bundle, "x") as tar:
        for name, path in sorted(files.items()):
            tar.add(path, arcname=name, recursive=False)
    with bundle.open("rb") as handle:
        os.fsync(handle.fileno())
    expected = {row["path"]: row for row in inventory}
    with tarfile.open(bundle) as tar:
        for member in tar:
            row = expected.pop(member.name)
            digest = hashlib.sha256()
            with tar.extractfile(member) as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if member.size != row["bytes"] or digest.hexdigest() != row["sha256"]:
                raise ValueError("archived unit payload differs")
    if expected:
        raise ValueError("archival unit incomplete")
    receipt = {
        "format": "speck_ordered_code_unit_archive",
        "format_version": 1,
        "unit_manifest_sha256": owner,
        "tar": {"path": str(bundle), "sha256": file_sha256(bundle), "bytes": bundle.stat().st_size},
        "inventory": {"path": str(inventory_path), "sha256": file_sha256(inventory_path)},
        "all_archival_payload_hashes_reopened": True,
    }
    durable_json(receipt_path, receipt)
    return receipt
