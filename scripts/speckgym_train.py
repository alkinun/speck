"""Train one baseline, procedural warm-up, or language phase from SpeckGym v0."""

import argparse

from scripts.base_train import train
from speck.config import load_experiment
from speck.speckgym import load_speckgym_config, resolve_training_phase


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", choices=tuple("ABCDE"))
    parser.add_argument("phase", choices=("warmup", "language"))
    parser.add_argument(
        "--experiment",
        default="experiments/SpeckGym-v0",
        help="SpeckGym experiment directory (default: %(default)s)",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", type=int, default=None)
    parser.add_argument("--no-compile", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    suite = load_speckgym_config(args.experiment)
    base_configs = load_experiment(suite["base_experiment"], "data", "tokenizer", "model", "train")
    configs = resolve_training_phase(suite, base_configs, args.run, args.phase)
    train(configs, args)


if __name__ == "__main__":
    main()
