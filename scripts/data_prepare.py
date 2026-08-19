"""prepare packed ultra-fineweb training and validation shards."""

import argparse

from speck.dataset import prepare_dataset


parser = argparse.ArgumentParser()
parser.add_argument("--train-tokens", type=int, default=10_000_524_288)
parser.add_argument("--validation-tokens", type=int, default=20_000_000)
parser.add_argument("--shard-tokens", type=int, default=100_000_000)
parser.add_argument("--validation-fraction", type=float, default=0.002)
parser.add_argument("--min-score", type=float, default=0.8)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output-dir", default=None)
args = parser.parse_args()

prepare_dataset(
    train_tokens=args.train_tokens,
    validation_tokens=args.validation_tokens,
    shard_tokens=args.shard_tokens,
    validation_fraction=args.validation_fraction,
    min_score=args.min_score,
    seed=args.seed,
    output_dir=args.output_dir,
)
