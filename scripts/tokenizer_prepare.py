"""Download and verify a Speck experiment tokenizer."""

import argparse

from speck.config import load_experiment
from speck.tokenizer import prepare


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment",
        nargs="?",
        default="experiments/Speck1-140M",
        help="experiment directory (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_experiment(args.experiment, "tokenizer")["tokenizer"]
    tokenizer = prepare(**config)
    print(f"Prepared tokenizer with {tokenizer.vocab_size:,} tokens")


if __name__ == "__main__":
    main()
