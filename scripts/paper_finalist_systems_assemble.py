"""Assemble one finalist systems trial or paired block from immutable artifacts."""

import argparse
import json
from pathlib import Path

from speck.paper_finalist_systems_assembly import assemble_block, assemble_trial
from speck.paper_finalist_systems_telemetry import atomic_json


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    trial = subparsers.add_parser("trial")
    trial.add_argument("protocol", type=Path)
    trial.add_argument("memory_supplement", type=Path)
    trial.add_argument("plan", type=Path)
    trial.add_argument("engine_result", type=Path)
    trial.add_argument("trace", type=Path)
    trial.add_argument("runtime_attestation", type=Path)
    trial.add_argument("--block", type=int, required=True)
    trial.add_argument("--position", type=int, choices=(0, 1), required=True)
    trial.add_argument("--output", type=Path, required=True)
    block = subparsers.add_parser("block")
    block.add_argument("protocol", type=Path)
    block.add_argument("--block", type=int, required=True)
    block.add_argument("trials", nargs="*", type=Path)
    block.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    if args.command == "trial":
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        matches = [
            trial
            for trial in plan.get("trials", ())
            if trial.get("block") == args.block and trial.get("position") == args.position
        ]
        if len(matches) != 1:
            raise ValueError("systems assembly trial is absent from the workload plan")
        result = assemble_trial(
            args.protocol,
            args.memory_supplement,
            matches[0],
            args.engine_result,
            args.trace,
            args.runtime_attestation,
        )
    else:
        result = assemble_block(args.protocol, args.block, args.trials)
    atomic_json(args.output, result)
    print(f"{result['format']}: {result['status']}")


if __name__ == "__main__":
    main()
