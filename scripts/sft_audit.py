"""Audit local SFT stock against a frozen tokenizer without downloading or training."""

import argparse
from pathlib import Path

from speck.provenance.io import atomic_json
from speck.tokenization.chat import ChatTokenizer
from speck.tokenization.tokenizer import Tokenizer
from speck.training.sft_audit import audit_sft


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="local .arrow or .parquet files")
    parser.add_argument("--tokenizer", required=True, help="frozen SentencePiece model file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples-per-subset", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096])
    args = parser.parse_args(argv)
    if Path(args.output).exists():
        raise FileExistsError(f"audit output already exists: {args.output}")
    report = audit_sft(
        args.inputs,
        ChatTokenizer(Tokenizer(args.tokenizer)),
        samples_per_subset=args.samples_per_subset,
        seed=args.seed,
        lengths=args.lengths,
        progress=lambda message: print(message, flush=True),
    )
    atomic_json(args.output, report)
    print(f"Audit: {args.output}")


if __name__ == "__main__":
    main()
