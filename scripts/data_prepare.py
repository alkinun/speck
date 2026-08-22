"""Prepare packed Ultra-FineWeb training and validation shards."""

import argparse

from speck.config import load_experiment
from speck.dataset import prepare_dataset
from speck.tokenizer import get_tokenizer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "experiment",
    nargs="?",
    default="experiments/Speck1-200M",
    help="experiment directory (default: %(default)s)",
)
parser.add_argument(
    "--restart",
    action="store_true",
    help="discard and replace an incomplete staged build",
)
args = parser.parse_args()

configs = load_experiment(args.experiment, "data", "tokenizer")
data = dict(configs["data"])
source = data.pop("source", None)
prepare_dataset(
    **data, source=source, tokenizer=get_tokenizer(**configs["tokenizer"]), restart=args.restart
)
