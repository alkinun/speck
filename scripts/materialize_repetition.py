"""Materialize a bound repetition source plan; does not create a model-launch manifest."""

import argparse
from pathlib import Path

from speck.data.repetition import materialize_repetition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = materialize_repetition(args.plan, resume=args.resume)
    print(result["status"])


if __name__ == "__main__":
    main()
