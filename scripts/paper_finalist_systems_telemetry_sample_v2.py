"""Acquire finite 1 Hz finalist systems telemetry with used-memory bytes."""

import argparse
from pathlib import Path

from speck.paper_finalist_systems_sampler_memory import serialize_memory_sample_series
from speck.paper_finalist_systems_telemetry import atomic_json


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("memory_supplement", type=Path)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--benchmark-pid", action="append", type=int, default=[])
    parser.add_argument("--disk-device", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    result = serialize_memory_sample_series(
        args.protocol,
        args.memory_supplement,
        args.samples,
        args.benchmark_pid,
        args.disk_device,
    )
    atomic_json(args.output, result)
    print(f"{result['format']}: {len(result['samples'])} memory-qualified samples")


if __name__ == "__main__":
    main()
