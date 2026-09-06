"""Acquire a finite 1 Hz finalist systems telemetry sample series."""

import argparse
from pathlib import Path

from speck.paper_finalist_systems_sampler import serialize_sample_series
from speck.paper_finalist_systems_telemetry import atomic_json


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--benchmark-pid", action="append", type=int, default=[])
    parser.add_argument("--disk-device", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    result = serialize_sample_series(
        args.protocol,
        args.samples,
        args.benchmark_pid,
        args.disk_device,
    )
    atomic_json(args.output, result)
    print(f"{result['format']}: {len(result['samples'])} samples")


if __name__ == "__main__":
    main()
