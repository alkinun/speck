"""Qualify bounded bank preparation and injected recovery from one clean checkout."""

import argparse
import json
import resource
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from speck.data.source_bank import load_bank_plan, prepare_source_bank
from speck.provenance.io import durable_json, file_sha256
from speck.provenance.repository import repository_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "plan", type=Path, help="Committed bounded plan with a new clean-bank destination"
    )
    parser.add_argument("report", type=Path, help="New compact qualification result")
    args = parser.parse_args()
    root = repository_root(__file__)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("source bank qualification requires a clean checkout")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if args.report.exists():
        raise FileExistsError(f"source bank qualification already exists: {args.report}")
    plan_path = args.plan.resolve()
    plan = load_bank_plan(plan_path)
    raw = json.loads(plan_path.read_text())
    runtime = Path(plan["output_directory"]).parent
    recovery_output = runtime / "resumed"
    recovery_plan_path = runtime / "resumed-plan.json"
    for path in (Path(plan["output_directory"]), recovery_output, recovery_plan_path):
        if path.exists():
            raise FileExistsError(f"qualification requires new destinations: {path}")
    raw["output_directory"] = str(recovery_output)
    # Resolve identities before moving the recovery plan to the runtime volume.
    raw["parent_manifest"] = plan["parent_manifest"]
    raw["reference_tokenizer"] = plan["reference_tokenizer"]
    runtime.mkdir(parents=True, exist_ok=True)
    durable_json(recovery_plan_path, raw)
    free_before = shutil.disk_usage(runtime).free
    clean = prepare_source_bank(plan_path)
    checkpoint_records = plan["checkpoint_records"]
    failure_started = time.perf_counter()
    try:
        prepare_source_bank(
            recovery_plan_path,
            interrupt_source="code",
            interrupt_after_records=checkpoint_records + 1,
        )
    except RuntimeError as error:
        if str(error) != "injected source bank selection interruption":
            raise
        interrupted_seconds = time.perf_counter() - failure_started
    else:
        raise RuntimeError("qualification did not reach the declared interruption")
    state = json.loads((recovery_output / "code/selection-state.json").read_text())
    if state["records"] != checkpoint_records:
        raise RuntimeError("qualification did not retain the expected recovery boundary")
    resumed = prepare_source_bank(recovery_plan_path)
    if clean["sources"] != resumed["sources"]:
        raise RuntimeError("source bank resumed/uninterrupted payload parity failed")
    reopened = prepare_source_bank(plan_path)
    if reopened["manifest"] != clean["manifest"]:
        raise RuntimeError("source bank published reopen identity failed")
    result = {
        "format": "speck_bounded_source_bank_qualification",
        "format_version": 1,
        "status": "six_category_byte_selection_reference_packing_and_resume_pass",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "repository_revision": revision,
        "plan": {"path": str(args.plan), "sha256": file_sha256(plan_path)},
        "recovery_plan": {
            "path": str(recovery_plan_path),
            "sha256": file_sha256(recovery_plan_path),
        },
        "clean": clean,
        "resumed": resumed,
        "reopened": reopened,
        "interruption": {
            "source": "code",
            "injected_after_records": checkpoint_records + 1,
            "recovered_from_records": state["records"],
            "attempt_elapsed_seconds": interrupted_seconds,
        },
        "resources": {
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "free_bytes_before": free_before,
            "free_bytes_after": shutil.disk_usage(runtime).free,
            "runtime_tree_bytes": sum(
                path.stat().st_size for path in runtime.rglob("*") if path.is_file()
            ),
        },
        "boundary": (
            "Bounded reuse of six retained pilot training sources with inherited filtering and firewall exclusions; "
            "byte quotas do not select D5 or E1/E3 treatments. Measured times cover identity verification, "
            "selection, reference-token packing and recovery, not upstream acquisition/global dedup, "
            "full E1/E3 capacity, production throughput, or useful long-document reconstruction. "
            "Free-space endpoints are snapshots, not high-water telemetry."
        ),
        "operations_authority": False,
        "training_authority": False,
    }
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as handle:
        handle.write(payload)
    print(
        json.dumps(
            {
                "status": result["status"],
                "report": str(args.report),
                "selected_utf8_bytes": sum(
                    item["selection"]["utf8_bytes"] for item in clean["sources"]
                ),
                "reference_tokens": sum(
                    item["packing"]["reference_tokens"] for item in clean["sources"]
                ),
                "clean_elapsed_seconds": clean["elapsed_seconds"],
                "interrupted_elapsed_seconds": interrupted_seconds,
                "resumed_elapsed_seconds": resumed["elapsed_seconds"],
                "reopen_elapsed_seconds": reopened["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
