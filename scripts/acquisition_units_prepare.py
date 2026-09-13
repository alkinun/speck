"""Prepare the frozen row-window acquisition units and retain per-unit attempt costs."""

import argparse
import json
from pathlib import Path

from speck.data.acquisition_units import load_unit_plan, prepare_units


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan")
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    result = prepare_units(load_unit_plan(args.plan), args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"Prepared {len(result['units'])} acquisition units; report: {args.report}")


if __name__ == "__main__":
    main()
