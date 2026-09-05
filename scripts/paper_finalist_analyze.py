"""Collect, target-lock, and analyze Paper 1 finalist evidence."""

import argparse
from pathlib import Path

from speck.paper_finalist_analysis import (
    analyze_finalist,
    atomic_json,
    collect_run_result,
    lock_time_to_quality_target,
)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect = subparsers.add_parser("collect")
    collect.add_argument("plan", type=Path)
    collect.add_argument("materialization_contract", type=Path)
    collect.add_argument("experiment", type=Path)
    collect.add_argument("--checkpoint-dir", type=Path, default=None)
    collect.add_argument("--output", type=Path, required=True)
    lock = subparsers.add_parser("lock-target")
    lock.add_argument("plan", type=Path)
    lock.add_argument("materialization_contract", type=Path)
    lock.add_argument("results", nargs="+", type=Path)
    lock.add_argument("--output", type=Path, required=True)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("plan", type=Path)
    analyze.add_argument("materialization_contract", type=Path)
    analyze.add_argument("results", nargs="+", type=Path)
    analyze.add_argument("--target-lock", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def run(args):
    if args.command == "collect":
        result = collect_run_result(
            args.plan,
            args.materialization_contract,
            args.experiment,
            args.checkpoint_dir,
        )
    elif args.command == "lock-target":
        result = lock_time_to_quality_target(
            args.plan, args.materialization_contract, args.results
        )
    else:
        result = analyze_finalist(
            args.plan,
            args.materialization_contract,
            args.target_lock,
            args.results,
        )
    atomic_json(args.output, result)
    return result


def main(argv=None):
    result = run(arguments(argv))
    print(f"{result['format']}: {result['status']}")


if __name__ == "__main__":
    main()
