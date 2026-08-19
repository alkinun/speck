"""download and verify an experiment tokenizer."""

import argparse

from speck.config import load_experiment
from speck.tokenizer import prepare


parser = argparse.ArgumentParser()
parser.add_argument("experiment", nargs="?", default="experiments/speck-50m")
args = parser.parse_args()

config = load_experiment(args.experiment, "tokenizer")["tokenizer"]
tokenizer = prepare(**config)
print(f"prepared tokenizer with {tokenizer.vocab_size:,} tokens")
