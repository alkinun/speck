"""Integrate one frozen-protocol finalist systems telemetry trace."""

import argparse
from pathlib import Path

from speck.paper_finalist_systems_telemetry import atomic_json, integrate_trace


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    result = integrate_trace(args.protocol, args.trace)
    atomic_json(args.output, result)
    print(
        f"{result['format']}: qualified={str(result['qualified']).lower()} "
        f"energy_joules={result['gross_board_energy_joules']['estimate']:.6f}"
    )


if __name__ == "__main__":
    main()
