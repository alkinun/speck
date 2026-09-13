"""Verify the research archive or restore its original runnable checkout."""

import argparse
import json
from pathlib import Path

from speck.provenance.archive import locate_original, restore_checkout, verify_archive


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "check", help="verify inventory, original Git identities, and archived bytes"
    )
    restore = commands.add_parser("restore", help="create a detached pre-cleanup checkout")
    restore.add_argument("destination", type=Path)
    locate = commands.add_parser(
        "locate", help="resolve an original path or a removed historical tool"
    )
    locate.add_argument("original_path")
    args = parser.parse_args(argv)
    if args.command == "check":
        print(json.dumps(verify_archive(), indent=2))
    elif args.command == "locate":
        print(json.dumps(locate_original(args.original_path), indent=2))
    else:
        print(restore_checkout(args.destination))


if __name__ == "__main__":
    main()
