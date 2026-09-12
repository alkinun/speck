"""Identify Python changes that require evidence successor qualification."""

import argparse
import json
from pathlib import Path

from speck.source_pins import changed_source_pins, source_pin_inventory


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--base", default="HEAD", help="Git tree containing the source pins")
    parser.add_argument("--target", default=None, help="optional Git tree to compare with the base")
    parser.add_argument(
        "--inventory",
        action="store_true",
        help="list every evidence-bound Python file in the base tree",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    return parser.parse_args(argv)


def run(args):
    if args.inventory and args.target is not None:
        raise ValueError("--inventory does not use --target")
    if args.inventory:
        entries = source_pin_inventory(args.repository, args.base)
        mode = "inventory"
    else:
        entries = changed_source_pins(args.repository, args.base, args.target)
        mode = "changed"
    report = {
        "format": "speck_source_pin_check",
        "format_version": 1,
        "mode": mode,
        "base": args.base,
        "target": args.target,
        "evidence_bound_files": len(entries),
        "files": entries,
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif mode == "inventory":
        print(f"{len(entries)} evidence-bound Python files at {args.base}")
        for entry in entries:
            print(f"{entry['path']} ({len(entry['references'])} references)")
    elif entries:
        print("Evidence-bound Python changes require an explicit successor qualification:")
        for entry in entries:
            print(f"- {entry['path']}")
            for reference in entry["references"]:
                print(f"  - {reference}")
    else:
        target = args.target or "the working tree"
        print(f"No evidence-bound Python changes between {args.base} and {target}.")
    return 1 if mode == "changed" and entries else 0


def main():
    raise SystemExit(run(arguments()))


if __name__ == "__main__":
    main()
