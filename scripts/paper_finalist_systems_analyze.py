"""Analyze completed blocks under the frozen Paper 1 finalist systems protocol."""

import argparse
from pathlib import Path

from speck.paper_finalist_systems_analysis import analyze_systems, atomic_json


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("results", nargs="*", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    result = analyze_systems(args.protocol, args.results)
    atomic_json(args.output, result)
    print(f"{result['format']}: {result['status']}")


if __name__ == "__main__":
    main()
