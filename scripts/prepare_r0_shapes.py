"""Expand and check the selected long-context R0 shapes without running GPU work."""

import argparse
from pathlib import Path

from speck.operations.r0_shapes import prepare_r0_shapes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = prepare_r0_shapes(args.plan, args.output)
    for case in result["cases"]:
        print(f"{case['id']}: {case['instantiated_parameters']:,} parameters; GPU checks pending")


if __name__ == "__main__":
    main()
