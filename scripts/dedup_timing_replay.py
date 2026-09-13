"""Measure phase-separated continuation from a private copy of the qualified reference checkpoint."""

import argparse
import json
import resource
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from speck.data.acquisition_units import _bound_identity
from speck.data.checkpoint_replay import restore_reference_checkpoint
from speck.data.firewall_integration import analyze_exclusion, run_exclusion
from speck.provenance.io import durable_json, file_sha256
from speck.provenance.repository import repository_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    root = repository_root(__file__)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("timing replay requires a clean implementation checkout")
    if args.report.exists():
        raise FileExistsError(args.report)
    path = args.plan.resolve()
    plan = json.loads(path.read_text())
    if (
        plan.get("format") != "speck_dedup_timing_replay_plan"
        or plan.get("format_version") != 2
        or plan.get("required_parity") != ["outputs", "removals", "counts"]
        or plan.get("training_authority") is not False
        or plan.get("restoration_durability")
        != "fsync copied committed prefixes and restored index before timing"
        or plan.get("checkpoint_components")
        != ["sqlite_commit", "output_flush_fsync", "slice_hash", "state_publication"]
    ):
        raise ValueError("unsupported dedup timing replay plan")
    parent_identity = _bound_identity(plan["parent_result"], path.parent)
    parent = json.loads(Path(parent_identity["path"]).read_text())
    output = Path(plan["output_directory"])
    config, restoration = restore_reference_checkpoint(parent, output)
    durable_json(output.parent / f"{output.name}-restoration.json", restoration)
    timing = {}
    try:
        replay = run_exclusion(config, timing=timing)
    except BaseException:
        durable_json(output.parent / f"{output.name}-incomplete-timing.json", timing)
        raise
    original = parent["exclusion"]["result"]["manifest"]
    for key in plan["required_parity"]:
        if replay["result"]["manifest"][key] != original[key]:
            raise RuntimeError(f"timing instrumentation/replay changed {key}")
    analysis = analyze_exclusion(output, parent["analysis"]["references"])
    candidates = [
        source for source in timing["sources"] if source["id"].startswith("acquired_train__")
    ]
    result = {
        "format": "speck_dedup_phase_timing_qualification",
        "format_version": 2,
        "status": "reference_checkpoint_replay_parity_and_phase_timing_pass",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "repository_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "plan": {"path": str(args.plan), "sha256": file_sha256(path)},
        "parent_result": parent_identity,
        "restoration": restoration,
        "timing": timing,
        "candidate_summary": {
            "records_seen": sum(source["records_seen"] for source in candidates),
            "processed_utf8_bytes": sum(source["processed_utf8_bytes"] for source in candidates),
            "processing_seconds_excluding_checkpoints": sum(
                source["processing_seconds_excluding_checkpoints"] for source in candidates
            ),
            "checkpoint_seconds": sum(source["checkpoint_seconds"] for source in candidates),
        },
        "analysis": analysis,
        "main_process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "boundary": "Timing-only continuation of the same qualified candidates. Private prefix restoration cost is separate; the reference index is not rebuilt. Pruning restores committed state, not an uncommitted-rollback stress case. Copying warms caches and retains an allocated index layout; these measurements are not cold-start or production-scale throughput. No new corpus capacity is added.",
        "operations_authority": False,
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
                "phases": timing["phases"],
                "candidates": result["candidate_summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
