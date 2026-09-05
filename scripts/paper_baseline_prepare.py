"""Materialize or verify the paired Speck Paper 1 baseline experiments."""

import argparse
import json
from pathlib import Path

from speck.paper_baseline import materialize_baselines


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    result = materialize_baselines(args.matrix, args.output_root, args.check)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
