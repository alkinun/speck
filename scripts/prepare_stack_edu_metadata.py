"""Acquire and verify the eleven complete metadata shards for matched code preparation."""

import argparse
import subprocess
from pathlib import Path

from speck.data.stack_edu_metadata import acquire_metadata, load_metadata_plan
from speck.provenance.io import durable_json
from speck.provenance.repository import repository_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = repository_root(__file__)
    if args.report.exists():
        raise FileExistsError(args.report)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("metadata acquisition requires a clean implementation")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    result = acquire_metadata(load_metadata_plan(args.plan), revision=revision, resume=args.resume)
    durable_json(args.report, result)
    print({"status": result["status"], "physical_rows": result["physical_rows"]}, flush=True)


if __name__ == "__main__":
    main()
