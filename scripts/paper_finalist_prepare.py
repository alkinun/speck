"""Materialize or verify the frozen Paper 1 finalist configs without launching them."""

import argparse
import json
from pathlib import Path

from speck.paper_finalist import materialize_finalist


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    result = materialize_finalist(args.contract, args.output_root, args.check)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
