"""Prepare a bounded byte-selected bank using retained firewall-excluded training inputs."""

import argparse
import json
from pathlib import Path

from speck.data.source_bank import prepare_source_bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan")
    parser.add_argument(
        "--report", required=True, type=Path, help="New per-invocation timing report"
    )
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(f"source bank timing report already exists: {args.report}")
    result = prepare_source_bank(args.plan)
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as handle:
        handle.write(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
