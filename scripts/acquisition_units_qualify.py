"""Qualify raw acquisition windows and cohort dedup with retained failure/timing records."""

import argparse
import cProfile
import json
import pstats
import resource
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from speck.data.acquisition_units import (
    _contamination_indexes,
    acquire_unit,
    deduplicate_units,
    load_unit_plan,
    prepare_units,
)
from speck.data.production_rehearsal import _logical_sqlite_identity
from speck.provenance.io import durable_json, file_sha256
from speck.provenance.repository import repository_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument(
        "runtime", type=Path, help="New qualification directory, separate from raw cache"
    )
    parser.add_argument("report", type=Path, help="New checked result JSON")
    args = parser.parse_args()
    root = repository_root(__file__)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("acquisition qualification requires a clean checkout")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if args.runtime.exists() or args.report.exists():
        raise FileExistsError("qualification requires fresh runtime and report paths")
    plan = load_unit_plan(args.plan)
    args.runtime.mkdir(parents=True)
    progress = {"repository_revision": revision, "events": []}

    def event(stage, result):
        progress["events"].append(
            {
                "stage": stage,
                "result": result,
                "free_bytes": shutil.disk_usage(args.runtime).free,
                "main_process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
            }
        )
        durable_json(args.runtime / "progress.json", progress)

    event("started", {"plan_sha256": file_sha256(args.plan)})
    try:
        acquired = args.runtime / "acquired"
        clean = prepare_units(plan, acquired)
        durable_json(args.runtime / "acquisition-report.json", clean)
        event("acquired", {"elapsed_seconds": clean["elapsed_seconds"]})
        for category in ("web", "code", "math", "synthetic", "science", "reference"):
            if not any(
                item["manifest"]["category"] == category
                and item["manifest"]["retained_records"] > 0
                for item in clean["units"]
            ):
                raise RuntimeError(f"acquisition has no retained records for {category}")
        if clean["units"][0]["manifest"]["retained_records"] == 0:
            raise RuntimeError("exact-replay control requires a nonempty first unit")
        contamination = _contamination_indexes(plan["base"])
        recovery = []
        for category in ("web", "code"):
            unit = next(item for item in plan["units"] if item["category"] == category)
            started = time.perf_counter()
            try:
                acquire_unit(
                    plan,
                    unit,
                    args.runtime / "recovery",
                    contamination,
                    interrupt_after_rows=plan["checkpoint_rows"] + 1,
                )
            except RuntimeError as error:
                if str(error) != "injected acquisition unit interruption":
                    raise
                failure_seconds = time.perf_counter() - started
                state = json.loads(
                    (args.runtime / "recovery" / unit["id"] / "state.json").read_text()
                )["state"]
                if state["yielded_rows"] != plan["checkpoint_rows"]:
                    raise RuntimeError(
                        "acquisition recovery boundary differs from the frozen interval"
                    )
                event(
                    f"{category}_interruption", {"elapsed_seconds": failure_seconds, "state": state}
                )
            else:
                raise RuntimeError("declared acquisition interruption was not reached")
            resumed = acquire_unit(plan, unit, args.runtime / "recovery", contamination)
            original = next(
                item for item in clean["units"] if item["manifest"]["unit_id"] == unit["id"]
            )
            if original["manifest"] != resumed["manifest"]:
                raise RuntimeError("acquisition recovered manifest/output parity failed")
            recovery.append(
                {"unit_id": unit["id"], "interrupted_seconds": failure_seconds, "resumed": resumed}
            )
            event(f"{category}_recovery", {"elapsed_seconds": resumed["elapsed_seconds"]})
        profiler = cProfile.Profile()
        dedup = profiler.runcall(deduplicate_units, plan, acquired, args.runtime / "dedup-clean")
        profiler.dump_stats(str(args.runtime / "dedup-profile.pstats"))
        durable_json(args.runtime / "dedup-report.json", dedup)
        event("dedup_clean_profiled", {"elapsed_seconds": dedup["elapsed_seconds"]})
        started = time.perf_counter()
        try:
            deduplicate_units(
                plan,
                acquired,
                args.runtime / "dedup-resumed",
                crash_after_records=plan["dedup_checkpoint_records"] + 1,
            )
        except RuntimeError as error:
            if str(error) != "injected production preprocess crash":
                raise
            dedup_interrupted_seconds = time.perf_counter() - started
            event("dedup_interruption", {"elapsed_seconds": dedup_interrupted_seconds})
        else:
            raise RuntimeError("declared dedup interruption was not reached")
        resumed_dedup = deduplicate_units(plan, acquired, args.runtime / "dedup-resumed")
        left, right = dedup["result"]["manifest"], resumed_dedup["result"]["manifest"]
        for key in ("counts", "outputs", "removals"):
            if left[key] != right[key]:
                raise RuntimeError(f"cohort dedup recovery {key} parity failed")
        logical = _logical_sqlite_identity(args.runtime / "dedup-clean" / left["index"]["path"])
        if logical != _logical_sqlite_identity(
            args.runtime / "dedup-resumed" / right["index"]["path"]
        ):
            raise RuntimeError("cohort dedup logical SQLite parity failed")
        if left["outputs"]["exact_replay_control"]["bytes"] != 0:
            raise RuntimeError("lower-precedence exact replay control was retained")
        event("dedup_recovery", {"elapsed_seconds": resumed_dedup["elapsed_seconds"]})
        profile = []
        for (filename, line, name), (primitive, calls, own, cumulative, _) in sorted(
            pstats.Stats(profiler).stats.items(), key=lambda item: item[1][3], reverse=True
        )[:30]:
            profile.append(
                {
                    "file": filename,
                    "line": line,
                    "function": name,
                    "calls": calls,
                    "primitive_calls": primitive,
                    "own_seconds": own,
                    "cumulative_seconds": cumulative,
                }
            )
        report = {
            "format": "speck_acquisition_unit_qualification",
            "format_version": 1,
            "status": "bounded_acquisition_and_ordered_cohort_dedup_recovery_pass",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "repository_revision": revision,
            "plan": {"path": str(args.plan), "sha256": file_sha256(args.plan)},
            "acquisition": clean,
            "acquisition_recovery": recovery,
            "deduplication": dedup,
            "deduplication_recovery": resumed_dedup,
            "dedup_interrupted_seconds": dedup_interrupted_seconds,
            "logical_index": logical,
            "profile": profile,
            "profile_boundary": "Clean dedup includes cProfile overhead; cumulative times overlap and must not be summed.",
            "progress": {
                "path": str(args.runtime / "progress.json"),
                "sha256": file_sha256(args.runtime / "progress.json"),
            },
            "boundary": (
                "Twelve physical row windows from six pinned raw files; inherited reader settings and benchmark/security filters "
                "are executed, followed by global dedup across this bounded cohort and one declared exact-replay control. "
                "Full tokenizer/selection/audit firewall reference exclusion is not run, so outputs cannot enter training or the "
                "qualified firewall-excluded source-bank path. This is not a production throughput or E1/E3 supply qualification."
            ),
            "operations_authority": False,
            "training_authority": False,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as handle:
            handle.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "report": str(args.report),
                    "acquisition_seconds": clean["elapsed_seconds"],
                    "dedup_profiled_seconds": dedup["elapsed_seconds"],
                    "counts": left["counts"],
                },
                indent=2,
            )
        )
    except BaseException as error:
        event("failed", {"type": type(error).__name__, "message": str(error)})
        raise


if __name__ == "__main__":
    main()
