"""Build one finalist systems runtime attestation from a qualified probe."""

import argparse
import json
from pathlib import Path

from speck.paper_finalist_systems_runtime import build_runtime_attestation
from speck.paper_finalist_systems_telemetry import atomic_json


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("trial_plan", type=Path)
    parser.add_argument("engine_result", type=Path)
    parser.add_argument("trace", type=Path)
    parser.add_argument("runtime_probe", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    trial = json.loads(args.trial_plan.read_text(encoding="utf-8"))
    result = build_runtime_attestation(
        args.protocol,
        trial,
        args.engine_result,
        args.trace,
        args.runtime_probe,
    )
    atomic_json(args.output, result)
    print(f"{result['format']}: {result['status']}")


if __name__ == "__main__":
    main()
