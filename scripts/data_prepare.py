"""Prepare source-separated packed training and validation shards."""

import argparse

from speck.config import load_experiment
from speck.dataset import prepare_dataset
from speck.tokenizer import get_tokenizer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment",
        nargs="?",
        default="experiments/Speck1-140M",
        help="experiment directory (default: %(default)s)",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="discard and replace an incomplete staged build",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    configs = load_experiment(args.experiment, "data", "tokenizer")
    prepare_dataset(
        **configs["data"],
        tokenizer=get_tokenizer(**configs["tokenizer"]),
        restart=args.restart,
    )


if __name__ == "__main__":
    main()
