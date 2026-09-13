"""Restore a retained reference checkpoint into a separate, identity-bound timing replay."""

import json
import os
import shutil
import sqlite3
import time
from pathlib import Path

from speck.data.production_data import (
    _verify_slices,
    accepted_document_chain,
    validate_preprocess_config,
)
from speck.provenance.io import durable_json, file_sha256


def _checked(identity):
    path = Path(identity["path"])
    if not path.is_file() or file_sha256(path) != identity["sha256"]:
        raise ValueError(f"timing replay input identity mismatch: {path}")
    return path


def _copy_prefix(source, destination, size):
    with source.open("rb") as original, destination.open("xb") as target:
        while size:
            block = original.read(min(size, 8 * 1024 * 1024))
            if not block:
                raise ValueError("timing replay source is shorter than its committed prefix")
            target.write(block)
            size -= len(block)
        target.flush()
        os.fsync(target.fileno())


def restore_reference_checkpoint(parent, output, *, sqlite_settings=None):
    """Copy verified prefix state and prune a private index copy; never modify the parent.

    Only the checkpoint contract is rebound to the successor destination/config,
    optionally including explicit SQLite settings. Reference state, input order,
    dedup policy, checkpoint cadence and candidate stream match the qualified pass.
    """

    started = time.perf_counter()
    if parent.get("status") != "complete_reference_exclusion_and_bank_handoff_pass":
        raise ValueError("timing replay requires a completed integration parent")
    manifest_path = _checked(parent["analysis"]["parent_manifest"])
    manifest = json.loads(manifest_path.read_text())
    if manifest != parent["exclusion"]["result"]["manifest"]:
        raise ValueError("timing replay parent manifest differs from its checked result")
    state_path = _checked(parent["interruption"]["retained_checkpoint"])
    state = json.loads(state_path.read_text())
    reference_records = parent["analysis"]["references"]["records"]
    if (
        state["contract"] != manifest["plan_fingerprint"]
        or state["source_index"] != 12
        or state["processed_records"] != reference_records
        or state["next_doc_seq"] != reference_records
    ):
        raise ValueError("timing replay is not the complete reference checkpoint")
    original = {
        "format": "speck_production_text_preprocess",
        "format_version": 1,
        "status": "fixture_or_rehearsal_authorized_not_training_authority",
        "sources": manifest["sources"],
        "deny_ledger": {key: manifest["deny_ledger"][key] for key in ("path", "sha256")},
        "policy": manifest["policy"],
        "checkpoint_records": 10000,
        "cleanup_files": manifest["cleanup_files"],
        "output_directory": str(manifest_path.parent),
    }
    if validate_preprocess_config(original)["plan_fingerprint"] != state["contract"]:
        raise ValueError("timing replay cannot reconstruct the original execution contract")
    output = Path(output).resolve()
    staging = output.with_name(output.name + ".building")
    if output.exists() or staging.exists():
        raise FileExistsError("timing replay requires a new destination")
    config = {**original, "output_directory": str(output)}
    if sqlite_settings is not None:
        config.update({"format_version": 2, "sqlite": sqlite_settings})
    normalized = validate_preprocess_config(config)
    staging.mkdir(parents=True)
    for index, source in enumerate(config["sources"]):
        entry = manifest["outputs"][source["id"]]
        destination = staging / f"{source['id']}.jsonl"
        size = state["output_sizes"].get(str(index), 0)
        _copy_prefix(manifest_path.parent / entry["path"], destination, size)
        _verify_slices(
            destination, state["output_slices"].get(str(index), []), size, "replay output"
        )
    removals = staging / "removals.jsonl"
    _copy_prefix(
        manifest_path.parent / manifest["removals"]["path"], removals, state["removal_size"]
    )
    _verify_slices(removals, state["removal_slices"], state["removal_size"], "replay removals")
    source_index = _checked(
        {
            "path": str(manifest_path.parent / manifest["index"]["path"]),
            "sha256": manifest["index"]["sha256"],
        }
    )
    wal = source_index.with_name(source_index.name + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError("timing replay requires a fully checkpointed parent SQLite file")
    index_path = staging / "near_duplicates.sqlite3"
    shutil.copyfile(source_index, index_path)
    if file_sha256(index_path) != manifest["index"]["sha256"]:
        raise ValueError("timing replay index copy differs from the parent")
    with index_path.open("rb") as handle:
        os.fsync(handle.fileno())
    connection = sqlite3.connect(index_path)
    try:
        # Work only in the private copy. Bulk-prune child rows first, then validate
        # referential integrity; this restoration is not a rollback stress benchmark.
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute("DELETE FROM bands WHERE doc_seq>=?", (state["next_doc_seq"],))
        connection.execute(
            "DELETE FROM docs WHERE processed_index>=?", (state["processed_records"],)
        )
        connection.execute(
            "DELETE FROM checkpoints WHERE checkpoint_id>?", (state["checkpoint_id"],)
        )
        connection.commit()
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise ValueError("timing replay index has dangling references")
        actual = accepted_document_chain(
            connection.execute("SELECT dedup_sha256, content_sha256 FROM docs ORDER BY doc_seq")
        )
        if actual != (state["next_doc_seq"], state["index_chain"]):
            raise ValueError("timing replay accepted-reference chain differs")
        checkpoint = connection.execute(
            "SELECT processed_records, next_doc_seq, index_chain FROM checkpoints WHERE checkpoint_id=?",
            (state["checkpoint_id"],),
        ).fetchone()
        if checkpoint != (state["processed_records"], state["next_doc_seq"], state["index_chain"]):
            raise ValueError("timing replay checkpoint row differs")
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        connection.close()
    with index_path.open("rb") as handle:
        os.fsync(handle.fileno())
    rebound_state = {**state, "contract": normalized["plan_fingerprint"]}
    durable_json(staging / "state.json", rebound_state)
    return config, {
        "parent_manifest": parent["analysis"]["parent_manifest"],
        "original_checkpoint": parent["interruption"]["retained_checkpoint"],
        "rebound_checkpoint_sha256": file_sha256(staging / "state.json"),
        "rebound_index_sha256": file_sha256(index_path),
        "changed_checkpoint_fields": ["contract"],
        **({"bound_sqlite": normalized["sqlite"]} if sqlite_settings is not None else {}),
        "restoration_durability": "committed prefix files and index fsynced before timing",
        "reference_records": reference_records,
        "restore_seconds": time.perf_counter() - started,
    }
