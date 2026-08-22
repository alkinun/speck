"""Prepare source-separated packed training and validation shards."""

import argparse

from speck.config import load_experiment
from speck.dataset import prepare_dataset
from speck.tokenizer import get_tokenizer

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
args = parser.parse_args()

configs = load_experiment(args.experiment, "data", "tokenizer")
prepare_dataset(
    **configs["data"],
    tokenizer=get_tokenizer(**configs["tokenizer"]),
    restart=args.restart,
)
