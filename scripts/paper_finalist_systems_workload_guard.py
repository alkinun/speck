"""Plan systems trials or run a command with protected-tree mutation detection."""

import argparse
from pathlib import Path

from speck.paper_finalist_systems_telemetry import atomic_json
from speck.paper_finalist_systems_workload import build_trial_plan, guarded_run


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command_name", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("protocol", type=Path)
    plan.add_argument("qualification", type=Path)
    plan.add_argument("--output-root", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    guard = subparsers.add_parser("guard")
    guard.add_argument("--protected", type=Path, action="append", required=True)
    guard.add_argument("--child-output", type=Path, required=True)
    guard.add_argument("--report", type=Path, required=True)
    guard.add_argument("child_command", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    if args.command_name == "plan":
        result = build_trial_plan(args.protocol, args.qualification, args.output_root)
        output = args.output
    else:
        child_command = (
            args.child_command[1:] if args.child_command[:1] == ["--"] else args.child_command
        )
        result = guarded_run(child_command, args.protected, args.child_output)
        output = args.report
    atomic_json(output, result)
    print(f"speck_paper_finalist_systems_workload_{args.command_name}: complete")


if __name__ == "__main__":
    main()
