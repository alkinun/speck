"""Download and verify a Speck experiment tokenizer."""

import argparse

from speck.config import load_experiment
from speck.tokenizer import prepare

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "experiment",
    nargs="?",
    default="experiments/Speck1-200M",
    help="experiment directory (default: %(default)s)",
)
args = parser.parse_args()

config = load_experiment(args.experiment, "tokenizer")["tokenizer"]
tokenizer = prepare(**config)
print(f"prepared tokenizer with {tokenizer.vocab_size:,} tokens")
