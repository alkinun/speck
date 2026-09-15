"""Prepare reusable restricted-code content units from complete files, without full exclusion."""

import argparse
import json
import os
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

from speck.data.production_rehearsal import _contamination_indexes
from speck.data.stack_v3_units import acquire_stack_v3_unit, load_stack_v3_acquisition, verify_raw
from speck.provenance.io import durable_json, file_sha256
from speck.provenance.repository import repository_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = repository_root(__file__)
    if (
        args.result.exists()
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    ):
        raise ValueError("requires new result and clean frozen implementation")
    if not os.path.ismount("/mnt/speck-data"):
        raise ValueError("data drive is not mounted")
    plan = load_stack_v3_acquisition(args.plan)
    output = Path(plan["output_directory"])
    execution = {
        "repository_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "plan": {"path": str(args.plan.resolve()), "sha256": file_sha256(args.plan)},
    }
    if output.exists():
        if not args.resume or json.loads((output / "execution.json").read_text()) != execution:
            raise ValueError("preserve earlier acquisition; resume requires identical execution")
    else:
        if args.resume:
            raise ValueError("cannot resume absent acquisition")
        output.mkdir(parents=True)
        durable_json(output / "execution.json", execution)
    started = time.perf_counter()
    for raw in plan["files"]:
        verify_raw(raw)
    print(f"verified all {len(plan['files'])} complete source files", flush=True)
    contamination = _contamination_indexes(plan["base"])
    reports = []
    by_language, reasons = {}, Counter()
    for unit in plan["units"]:
        if shutil.disk_usage(output).free < plan["minimum_free_bytes"]:
            raise ValueError("minimum free-space floor failed")
        used = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        remaining = plan["maximum_acquisition_output_bytes"] - used - 16777216
        if remaining <= 0:
            raise ValueError("acquisition output ceiling reached; preserve all attempts")
        report = acquire_stack_v3_unit(
            plan, unit, output / "acquired", contamination, remaining_bytes=remaining
        )
        reports.append(report)
        manifest = report["manifest"]
        reasons.update(manifest["rejections"])
        for language, values in manifest["language_counts_before_full_exclusion"].items():
            totals = by_language.setdefault(language, {"documents": 0, "tokens": 0})
            for key in totals:
                totals[key] += values[key]
        progress = {
            "complete_units": len(reports),
            "total_units": len(plan["units"]),
            "physical_files_seen": sum(item["manifest"]["physical_files_seen"] for item in reports),
            "retained_records": sum(row["documents"] for row in by_language.values()),
            "tokens_before_full_exclusion": sum(row["tokens"] for row in by_language.values()),
            "by_language_before_full_exclusion": by_language,
            "elapsed_seconds_this_invocation": time.perf_counter() - started,
            "full_exclusion_performed": False,
        }
        durable_json(output / "progress.json", progress)
        print(
            json.dumps(
                {
                    key: progress[key]
                    for key in (
                        "complete_units",
                        "total_units",
                        "retained_records",
                        "tokens_before_full_exclusion",
                    )
                }
            ),
            flush=True,
        )
    durable_json(
        args.result,
        {
            "format": "speck_stack_v3_stock_acquisition_result",
            "format_version": 1,
            "status": "content_units_verified_reference_and_candidate_exclusion_pending",
            **execution,
            "source_use": plan["source_use"]["identity"],
            "files": plan["files"],
            "units": reports,
            "progress": progress,
            "rejections": dict(reasons),
            "acquired_directory": str(output / "acquired"),
            "training_authority": False,
            "boundary": plan["scope"],
        },
    )
    print("complete content acquisition; full exclusion remains pending", flush=True)


if __name__ == "__main__":
    main()
