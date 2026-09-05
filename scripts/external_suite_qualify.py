"""Verify pinned commits and file hashes for an external evaluation suite."""

import argparse
import json
from pathlib import Path

from speck.external import qualify_external_suite


def checkout(value):
    try:
        identifier, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("checkouts must use id=path") from error
    if not identifier or not path:
        raise argparse.ArgumentTypeError("checkouts must use non-empty id=path")
    return identifier, Path(path)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--checkout", action="append", type=checkout, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    checkouts = dict(args.checkout)
    if len(checkouts) != len(args.checkout):
        raise ValueError("checkout ids must not be repeated")
    print(json.dumps(qualify_external_suite(args.config, checkouts), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
