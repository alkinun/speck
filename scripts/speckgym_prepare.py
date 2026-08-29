"""Prepare the B-E procedural corpora for the SpeckGym v0 experiment."""

import argparse
from pathlib import Path

from speck.config import load_experiment
from speck.speckgym import load_speckgym_config, prepare_speckgym
from speck.tokenizer import get_tokenizer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment",
        nargs="?",
        default="experiments/SpeckGym-v0",
        help="SpeckGym experiment directory (default: %(default)s)",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--restart",
        action="store_true",
        help="discard and replace incomplete staged procedural builds",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_speckgym_config(args.experiment)
    tokenizer_config = load_experiment(config["base_experiment"], "tokenizer")["tokenizer"]
    manifests = prepare_speckgym(
        config,
        get_tokenizer(**tokenizer_config),
        output_dir=args.output_dir,
        restart=args.restart,
    )
    for run, manifest in sorted(manifests.items()):
        print(f"{run}: {manifest['splits']['train']['tokens']:,} prepared train tokens")


if __name__ == "__main__":
    main()
