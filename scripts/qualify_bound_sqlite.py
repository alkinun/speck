"""Qualify config-bound SQLite policy through the normal preparation CLI and crash recovery."""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from speck.data.acquisition_units import _bound_identity
from speck.data.checkpoint_replay import restore_reference_checkpoint
from speck.data.preparation_policy import load_preparation_policy
from speck.data.production_rehearsal import _logical_sqlite_identity
from speck.data.sqlite_settings import verify_sqlite_runtime
from speck.data.sqlite_wal import CRASH_EXIT_CODE
from speck.provenance.io import durable_json, file_sha256
from speck.provenance.repository import repository_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path)
    parser.add_argument("runtime", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    root = repository_root(__file__)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("bound-policy qualification requires a clean checkout")
    if args.runtime.exists() or args.report.exists():
        raise FileExistsError("bound-policy qualification requires new output paths")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    policy = load_preparation_policy(args.policy)
    comparison = json.loads(Path(policy["qualification"]["path"]).read_text())
    parent_identity = _bound_identity(comparison["parent_result"], root)
    parent = json.loads(Path(parent_identity["path"]).read_text())
    args.runtime.mkdir(parents=True)
    config, restoration = restore_reference_checkpoint(
        parent, args.runtime / "prepared", sqlite_settings=policy["sqlite"]
    )
    config_path = args.runtime / "preprocess.json"
    durable_json(config_path, config)
    durable_json(
        args.runtime / "binding.json",
        {
            "policy": policy,
            "restoration": restoration,
            "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        },
    )
    receipt_path = args.runtime / "crash-receipt.json"
    worker_path = args.runtime / "worker.json"
    durable_json(
        worker_path,
        {
            "config": config,
            "pages": policy["sqlite"]["wal_autocheckpoint_pages"],
            "receipt": str(receipt_path),
        },
    )
    started = time.perf_counter()
    child = subprocess.run(
        [sys.executable, "-m", "scripts.dedup_wal_compare", "--crash-worker", str(worker_path)],
        cwd=root,
        check=False,
    )
    interrupted_seconds = time.perf_counter() - started
    if child.returncode != CRASH_EXIT_CODE:
        raise RuntimeError(f"bound-policy crash worker exited unexpectedly: {child.returncode}")
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt["policy_source"] != "bound_config"
        or receipt["main_file_documents_without_wal"] >= receipt["committed_documents"]
    ):
        raise RuntimeError("crash did not exercise the config-bound committed WAL")
    checkpoint = Path(receipt["checkpoint"]["path"])
    if file_sha256(checkpoint) != receipt["checkpoint"]["sha256"]:
        raise RuntimeError("bound-policy crash checkpoint changed")
    snapshot = args.runtime / "crash-checkpoint.json"
    durable_json(snapshot, json.loads(checkpoint.read_text()))
    if file_sha256(snapshot) != receipt["checkpoint"]["sha256"]:
        raise RuntimeError("retained crash checkpoint differs from its receipt")
    command = [sys.executable, "-m", "scripts.production_data_preprocess_batched", str(config_path)]
    started = time.perf_counter()
    subprocess.run(command, cwd=root, check=True)
    recovered_seconds = time.perf_counter() - started
    manifest_path = args.runtime / "prepared/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["format_version"] != 2 or manifest["sqlite"] != policy["sqlite"]:
        raise RuntimeError("ordinary CLI did not publish the bound v2 declaration")
    verify_sqlite_runtime(policy["sqlite"], manifest["sqlite_runtime"])
    original = parent["exclusion"]["result"]["manifest"]
    for key in ("outputs", "removals", "counts"):
        if manifest[key] != original[key]:
            raise RuntimeError(f"config-bound execution changed {key}")
    original_dir = Path(parent["analysis"]["parent_manifest"]["path"]).parent
    if _logical_sqlite_identity(
        manifest_path.parent / manifest["index"]["path"]
    ) != _logical_sqlite_identity(original_dir / original["index"]["path"]):
        raise RuntimeError("config-bound execution changed logical SQLite tables")
    digest = file_sha256(manifest_path)
    started = time.perf_counter()
    subprocess.run(command, cwd=root, check=True)
    reopen_seconds = time.perf_counter() - started
    if file_sha256(manifest_path) != digest:
        raise RuntimeError("ordinary CLI reopen changed the published manifest")
    result = {
        "format": "speck_bound_sqlite_policy_qualification",
        "format_version": 1,
        "status": "config_bound_policy_normal_cli_hard_crash_recovery_and_reopen_pass",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "repository_revision": revision,
        "policy": policy,
        "parent_result": parent_identity,
        "restoration": restoration,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "manifest": {"path": str(manifest_path), "sha256": digest},
        "sqlite_runtime": manifest["sqlite_runtime"],
        "crash": receipt,
        "retained_checkpoint": {"path": str(snapshot), "sha256": file_sha256(snapshot)},
        "timings": {
            "interrupted_seconds": interrupted_seconds,
            "recovered_cli_seconds": recovered_seconds,
            "reopen_cli_seconds": reopen_seconds,
        },
        "boundary": "Binding and recovery qualification through config v2 and the ordinary CLI; not another speed comparison or production-scale/storage-envelope qualification. V1 configs retain their existing identities and behavior.",
        "training_authority": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "report": str(args.report),
                "sqlite": manifest["sqlite_runtime"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
