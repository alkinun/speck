"""Compare durable WAL checkpoint policies with paired order reversal and hard-crash recovery."""

import argparse
import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from speck.data.acquisition_units import _bound_identity
from speck.data.checkpoint_replay import restore_reference_checkpoint
from speck.data.firewall_integration import analyze_exclusion, run_exclusion
from speck.data.production_rehearsal import _logical_sqlite_identity
from speck.data.sqlite_wal import CRASH_EXIT_CODE, assess_wal_comparison, wal_policy
from speck.provenance.io import durable_json, file_sha256
from speck.provenance.repository import repository_root


def crash_worker(path):
    spec = json.loads(Path(path).read_text())
    policy = {}
    with wal_policy(spec["pages"], policy, crash_receipt=spec["receipt"]):
        run_exclusion(spec["config"])
    raise RuntimeError("no candidate checkpoint required outstanding committed WAL")


def verify_run(config, result, original, logical):
    manifest = result["result"]["manifest"]
    for key in ("outputs", "removals", "counts"):
        if manifest[key] != original[key]:
            raise RuntimeError(f"WAL policy changed {key}")
    output = Path(config["output_directory"])
    if _logical_sqlite_identity(output / manifest["index"]["path"]) != logical:
        raise RuntimeError("WAL policy changed logical SQLite tables")
    wal = output / (manifest["index"]["path"] + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise RuntimeError("final publication left a nonempty WAL")
    return {"path": str(output / "manifest.json"), "sha256": file_sha256(output / "manifest.json")}


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--crash-worker":
        crash_worker(sys.argv[2])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    root = repository_root(__file__)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("WAL qualification requires a clean checkout")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    path = args.plan.resolve()
    plan = json.loads(path.read_text())
    expected_order = [
        {"id": "baseline_1", "wal_autocheckpoint_pages": 1000},
        {"id": "candidate_1", "wal_autocheckpoint_pages": 65536},
        {"id": "candidate_2", "wal_autocheckpoint_pages": 65536},
        {"id": "baseline_2", "wal_autocheckpoint_pages": 1000},
    ]
    if (
        plan.get("format") != "speck_sqlite_wal_comparison_plan"
        or plan.get("format_version") != 1
        or plan.get("order") != expected_order
        or plan.get("pairs") != [["baseline_1", "candidate_1"], ["baseline_2", "candidate_2"]]
        or plan.get("synchronous") != "FULL"
        or plan.get("page_size") != 4096
        or plan.get("checkpoint_records") != 10000
        or plan.get("minimum_relative_reduction") != 0.1
        or plan.get("maximum_observed_wal_bytes") != 536870912
        or plan.get("required_parity") != ["outputs", "removals", "counts", "logical_sqlite_tables"]
        or plan.get("production_default_change") is not False
        or plan.get("training_authority") is not False
    ):
        raise ValueError("WAL comparison differs from the frozen contract")
    output = Path(plan["output_directory"])
    if output.exists() or args.report.exists():
        raise FileExistsError("WAL qualification requires fresh runtime and report paths")
    parent_identity = _bound_identity(plan["parent_result"], path.parent)
    parent = json.loads(Path(parent_identity["path"]).read_text())
    original = parent["exclusion"]["result"]["manifest"]
    original_root = Path(parent["analysis"]["parent_manifest"]["path"]).parent
    logical = _logical_sqlite_identity(original_root / original["index"]["path"])
    output.mkdir(parents=True)
    progress = {"repository_revision": revision, "events": []}

    def event(stage, result):
        progress["events"].append(
            {"stage": stage, "at": datetime.now(timezone.utc).isoformat(), "result": result}
        )
        durable_json(output / "progress.json", progress)
        print(f"{stage}: {json.dumps(result, sort_keys=True)}", flush=True)

    runs = []
    try:
        for declaration in plan["order"]:
            name = declaration["id"]
            config, restoration = restore_reference_checkpoint(parent, output / name)
            durable_json(output / f"{name}-restoration.json", restoration)
            event(f"{name}_restored", {"seconds": restoration["restore_seconds"]})
            policy, timing = {}, {}
            with wal_policy(declaration["wal_autocheckpoint_pages"], policy):
                result = run_exclusion(config, timing=timing)
            manifest_identity = verify_run(config, result, original, logical)
            record = {
                "id": name,
                "restoration": restoration,
                "timing": timing,
                "policy": policy,
                "manifest": manifest_identity,
                "parity_pass": True,
            }
            durable_json(output / f"{name}-result.json", record)
            runs.append(record)
            event(
                name,
                {
                    "total_seconds": timing["total_seconds"],
                    "observed_peak_wal_bytes": policy["observed_peak_wal_bytes"],
                },
            )
        config, restoration = restore_reference_checkpoint(parent, output / "hard-crash")
        receipt_path = output / "hard-crash-receipt.json"
        worker_path = output / "hard-crash-worker.json"
        durable_json(worker_path, {"config": config, "pages": 65536, "receipt": str(receipt_path)})
        event("hard_crash_restored", {"seconds": restoration["restore_seconds"]})
        child = subprocess.run(
            [sys.executable, "-m", "scripts.dedup_wal_compare", "--crash-worker", str(worker_path)],
            cwd=root,
            check=False,
        )
        if child.returncode != CRASH_EXIT_CODE or not receipt_path.is_file():
            raise RuntimeError(
                f"hard-crash worker did not reach its declared exit: {child.returncode}"
            )
        receipt = json.loads(receipt_path.read_text())
        if receipt["main_file_documents_without_wal"] >= receipt["committed_documents"]:
            raise RuntimeError("crash qualification did not require committed WAL")
        checkpoint = receipt["checkpoint"]
        if file_sha256(checkpoint["path"]) != checkpoint["sha256"]:
            raise RuntimeError("crash checkpoint identity changed")
        durable_json(
            output / "hard-crash-checkpoint.json", json.loads(Path(checkpoint["path"]).read_text())
        )
        event(
            "hard_exit_observed",
            {
                "exit_code": child.returncode,
                "main_documents": receipt["main_file_documents_without_wal"],
                "committed_documents": receipt["committed_documents"],
                "wal_bytes": receipt["wal_bytes_at_crash"],
            },
        )
        recovery_policy, recovery_timing = {}, {}
        with wal_policy(65536, recovery_policy):
            recovered = run_exclusion(config, timing=recovery_timing)
        recovered_manifest = verify_run(config, recovered, original, logical)
        exclusion = analyze_exclusion(config["output_directory"], parent["analysis"]["references"])
        recovery_peak = max(
            receipt["wal_bytes_at_crash"], recovery_policy["observed_peak_wal_bytes"]
        )
        recovery = {
            "restoration": restoration,
            "receipt": receipt,
            "policy": recovery_policy,
            "timing": recovery_timing,
            "manifest": recovered_manifest,
            "parity_pass": True,
            "observed_peak_wal_bytes": recovery_peak,
            "space_gate_pass": recovery_peak <= plan["maximum_observed_wal_bytes"],
            "exact_reference_overlap": exclusion["exact_reference_overlap"],
            "controls": exclusion["controls"],
        }
        durable_json(output / "hard-crash-result.json", recovery)
        analysis = assess_wal_comparison(plan, runs, recovery["space_gate_pass"])
        event("complete", analysis)
        report = {
            "format": "speck_sqlite_wal_comparison_result",
            "format_version": 1,
            "status": "bounded_wal_policy_comparison_and_hard_crash_recovery_complete",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "repository_revision": revision,
            "sqlite_version": sqlite3.sqlite_version,
            "plan": {"path": str(args.plan), "sha256": file_sha256(path)},
            "parent_result": parent_identity,
            "parent_logical_index": logical,
            "runs": runs,
            "recovery": recovery,
            "analysis": analysis,
            "progress": {
                "path": str(output / "progress.json"),
                "sha256": file_sha256(output / "progress.json"),
            },
            "boundary": "Same qualified candidate stream and private durable reference-prefix restores. Complete timings include final WAL truncation/publication/verification; restore and independent parity checks are separate. WAL sizes are checkpoint-boundary observations, not a hard global cap. Hard process exit is not a physical power-loss test. Two paired local runs do not establish production-scale or GH200-site throughput.",
            "production_default_changed": False,
            "training_authority": False,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as handle:
            handle.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    except BaseException as error:
        event("failed", {"type": type(error).__name__, "message": str(error)})
        raise


if __name__ == "__main__":
    main()
