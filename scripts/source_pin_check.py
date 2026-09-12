"""Identify Python changes that require evidence successor qualification."""

import argparse
import json
from pathlib import Path

from speck.source_pins import changed_evidence_pins, evidence_pin_inventory, source_pin_inventory


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--base", default="HEAD", help="Git tree containing the source pins")
    parser.add_argument("--target", default=None, help="optional Git tree to compare with the base")
    parser.add_argument(
        "--inventory",
        action="store_true",
        help="list evidence-bound files in the base tree instead of checking changes",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="include non-Python files in inventory output; changed-file checks always include them",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    return parser.parse_args(argv)


def run(args):
    if args.inventory and args.target is not None:
        raise ValueError("--inventory does not use --target")
    if args.inventory:
        inventory = evidence_pin_inventory if args.all_files else source_pin_inventory
        entries = inventory(args.repository, args.base)
        mode = "inventory"
    else:
        entries = changed_evidence_pins(args.repository, args.base, args.target)
        mode = "changed"
    report = {
        "format": "speck_source_pin_check",
        "format_version": 1,
        "mode": mode,
        "scope": "all" if mode == "changed" or args.all_files else "python",
        "base": args.base,
        "target": args.target,
        "evidence_bound_files": len(entries),
        "files": entries,
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif mode == "inventory":
        scope = "files" if args.all_files else "Python files"
        print(f"{len(entries)} evidence-bound {scope} at {args.base}")
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
        print(f"No evidence-bound tracked-file changes between {args.base} and {target}.")
    return 1 if mode == "changed" and entries else 0


def main():
    raise SystemExit(run(arguments()))


if __name__ == "__main__":
    main()
