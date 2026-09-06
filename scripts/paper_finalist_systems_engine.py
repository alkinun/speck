"""Run one activation-gated non-persisting finalist systems trial."""

import argparse
from pathlib import Path

from speck.paper_finalist_systems_engine import run_trial
from speck.paper_finalist_systems_telemetry import atomic_json


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("qualification", type=Path)
    parser.add_argument("activation", type=Path)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--position", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    result = run_trial(
        args.protocol,
        args.qualification,
        args.activation,
        args.block,
        args.position,
        args.output,
    )
    atomic_json(args.output, result)
    print(f"speck_paper_finalist_systems_engine: {result['status']}")


if __name__ == "__main__":
    main()
