"""Collect, target-lock, and analyze Speck Paper 1 baseline evidence."""

import argparse
from pathlib import Path

from speck.paper_baseline_analysis import (
    analyze_baselines,
    atomic_json,
    collect_run_result,
    lock_time_to_quality_target,
)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="qualify one complete baseline run")
    collect.add_argument("plan", type=Path)
    collect.add_argument("experiment", type=Path)
    collect.add_argument("--checkpoint-dir", type=Path, default=None)
    collect.add_argument("--output", type=Path, required=True)

    lock = subparsers.add_parser(
        "lock-target", help="lock time-to-quality from the three dense controls"
    )
    lock.add_argument("plan", type=Path)
    lock.add_argument("results", nargs="+", type=Path)
    lock.add_argument("--output", type=Path, required=True)

    analyze = subparsers.add_parser("analyze", help="analyze all six frozen baseline runs")
    analyze.add_argument("plan", type=Path)
    analyze.add_argument("results", nargs="+", type=Path)
    analyze.add_argument("--target-lock", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def run(args):
    if args.command == "collect":
        result = collect_run_result(args.plan, args.experiment, args.checkpoint_dir)
    elif args.command == "lock-target":
        result = lock_time_to_quality_target(args.plan, args.results)
    else:
        result = analyze_baselines(args.plan, args.target_lock, args.results)
    atomic_json(args.output, result)
    return result


def main(argv=None):
    args = arguments(argv)
    result = run(args)
    print(f"{result['format']}: {result['status']}")


if __name__ == "__main__":
    main()
