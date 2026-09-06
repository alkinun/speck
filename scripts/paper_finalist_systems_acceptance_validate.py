"""Validate six assembled finalist systems blocks and their global identities."""

import argparse
from pathlib import Path

from speck.paper_finalist_systems_acceptance import validate_systems_evidence


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("blocks", nargs="*", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    report = validate_systems_evidence(args.protocol, args.blocks)
    print(
        "Finalist systems acceptance: "
        f"{report['status']} ({len(report['valid_complete_blocks'])} complete, "
        f"{len(report['failed_blocks'])} failed, {len(report['missing_blocks'])} missing)"
    )


if __name__ == "__main__":
    main()
