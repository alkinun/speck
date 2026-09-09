"""Validate and summarize the Speck research compendium."""

import argparse
import json
from pathlib import Path

from speck.research_catalog import validate_research_catalog


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "catalog",
        type=Path,
        nargs="?",
        default=Path("research/catalog.json"),
        help="research catalog JSON (default: research/catalog.json)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    print(json.dumps(validate_research_catalog(args.catalog), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
