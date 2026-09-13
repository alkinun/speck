"""Inspect frozen UltraData subsets for content, language, serialization and provenance."""

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

from speck.data.ultradata_intake import inspect_view
from speck.provenance.io import durable_json, file_sha256
from speck.provenance.repository import repository_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    root = repository_root(__file__)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("intake review requires a clean identified implementation")
    plan = json.loads(args.plan.read_text())
    if (
        plan.get("format") != "speck_ultradata_intake_plan"
        or plan.get("format_version") != 1
        or plan.get("purpose") != "bounded_content_and_metadata_review_not_training_data"
        or plan.get("rows_per_window") != 32
        or plan.get("windows") != ["first", "middle", "last"]
        or plan.get("maximum_response_bytes") != 33554432
        or plan.get("training_authority") is not False
    ):
        raise ValueError("unsupported bounded intake plan")
    output = Path(plan["output_directory"])
    views = plan.get("views", [])
    if (
        len(views) != 7
        or len({view["id"] for view in views}) != 7
        or any(
            not re.fullmatch(r"[a-z][a-z0-9_]+", view["id"])
            or not re.fullmatch(r"[0-9a-f]{40}", view["revision"])
            or view["kind"] not in ("python_source", "code_exercise", "math_text")
            for view in views
        )
        or plan["language_diagnostic"]
        != {
            "minimum_alphabetic_characters": 80,
            "minimum_english_probability": 0.8,
            "maximum_characters": 50000,
        }
    ):
        raise ValueError("intake view identities or language diagnostic differ from the plan")
    if output.exists() or args.report.exists():
        raise FileExistsError("intake destinations must be new")
    output.mkdir(parents=True)
    results, identifiers = [], {}
    for view in plan["views"]:
        try:
            result, ids = inspect_view(plan, view)
            identifiers[view["id"]] = ids
        except Exception as error:
            result = {
                "view": view,
                "status": "intake_blocked",
                "error_type": type(error).__name__,
                "error": str(error),
                "training_authority": False,
            }
            durable_json(output / view["id"] / "failure.json", result)
        results.append(result)
        print(
            json.dumps(
                {
                    "view": view["id"],
                    "status": result["status"],
                    "counts": result.get("summary", {}).get("counts", {}),
                }
            ),
            flush=True,
        )
    overlaps = [
        {
            "left": left,
            "right": right,
            "equal_sample_ids": len(identifiers[left] & identifiers[right]),
        }
        for left, right in combinations(identifiers, 2)
    ]
    result = {
        "format": "speck_ultradata_bounded_intake",
        "format_version": 1,
        "status": "all_requested_views_inspected"
        if all(row["status"].endswith("complete") for row in results)
        else "some_requested_views_blocked",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "repository_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "plan": {"path": str(args.plan), "sha256": file_sha256(args.plan)},
        "views": results,
        "sample_identifier_overlaps": overlaps,
        "boundary": "First/middle/last dataset-viewer windows with matching x-revision headers and retained response hashes. Not uniform corpus sampling, complete-shard SHA verification, correctness testing, contamination qualification, or training/source-use approval. Zero ID overlap in these windows does not establish unrelated ancestry.",
        "training_authority": False,
    }
    durable_json(args.report, result)


if __name__ == "__main__":
    main()
