"""prepare packed ultra-fineweb training and validation shards."""

import argparse

from speck.config import load_experiment
from speck.dataset import prepare_dataset
from speck.tokenizer import get_tokenizer


parser = argparse.ArgumentParser()
parser.add_argument("experiment", nargs="?", default="experiments/speck-50m")
parser.add_argument("--restart", action="store_true")
args = parser.parse_args()

configs = load_experiment(args.experiment, "data", "tokenizer")
data = dict(configs["data"])
source = data.pop("source", None)
prepare_dataset(**data, source=source, tokenizer=get_tokenizer(**configs["tokenizer"]), restart=args.restart)
