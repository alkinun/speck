"""Validate and summarize the Speck research compendium."""

import argparse
import json
from pathlib import Path

from speck.provenance.catalog import validate_research_catalog


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "catalog",
        type=Path,
        nargs="?",
        default=Path("research/catalog.json"),
        help="research catalog JSON (default: research/catalog.json)",
    )
    parser.add_argument("--status", action="store_true", help="show current state and next actions")
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    report = validate_research_catalog(args.catalog)
    if args.status:
        from speck.provenance.repository import repository_root

        status = json.loads((repository_root(args.catalog) / report["status_record"]).read_text())
        print(f"{status['program']} — {status['phase']} — {status['as_of']}")
        for work in status["work"]:
            print(f"\n{work['id']}: {work['state']}\n  {work['next']}")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
