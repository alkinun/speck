"""Build a deterministic, category-balanced tokenizer train/evaluation sample."""

import argparse

from speck.tokenizer_experiment import load_experiment_config, prepare_sample


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="tokenizer experiment JSON")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="discard and replace an incomplete sample build",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_experiment_config(args.config)
    manifest = prepare_sample(config, restart=args.restart)
    train_bytes = sum(
        category["splits"]["train"]["utf8_bytes"] for category in manifest["categories"]
    )
    evaluation_bytes = sum(
        category["splits"]["eval"]["utf8_bytes"] for category in manifest["categories"]
    )
    print(
        f"Prepared tokenizer sample with {train_bytes:,} training and "
        f"{evaluation_bytes:,} evaluation bytes"
    )


if __name__ == "__main__":
    main()
