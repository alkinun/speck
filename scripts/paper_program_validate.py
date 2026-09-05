"""Validate the preregistered Speck Paper 1 research program."""

import argparse
import json
from pathlib import Path

from speck.paper import validate_paper_program


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    print(json.dumps(validate_paper_program(args.directory), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
