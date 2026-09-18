"""Build an offline corpus census and stratified content-review packet outside the repository."""

import argparse
import json
from pathlib import Path

from speck.data.corpus_audit import build_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build_audit(json.loads(args.plan.read_text()), args.output)


if __name__ == "__main__":
    main()
