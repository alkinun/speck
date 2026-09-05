"""Validate a versioned Speck architecture-promotion contract."""

import argparse
import json
from pathlib import Path

from speck.config import load_experiment
from speck.research import validate_research_contract
from speck.tokenizer import get_tokenizer


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--tokenizer-experiment",
        type=Path,
        default=None,
        help="also qualify declared route strings against this experiment's prepared tokenizer",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    tokenizer = None
    if args.tokenizer_experiment is not None:
        config = load_experiment(args.tokenizer_experiment, "tokenizer")["tokenizer"]
        tokenizer = get_tokenizer(**config)
    print(
        json.dumps(
            validate_research_contract(args.directory, tokenizer=tokenizer),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
