"""Prepare packed assistant-masked chat data for instruction tuning."""

import argparse

from speck.config import load_experiment
from speck.tokenization.chat import get_chat_tokenizer
from speck.training.sft_data import prepare_sft_dataset


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment",
        help="experiment directory",
    )
    parser.add_argument("--restart", action="store_true", help="replace an incomplete staged build")
    parser.add_argument(
        "--source-dir", help="directory containing pinned local conversation Parquet files"
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    configs = load_experiment(args.experiment, "tokenizer", "sft")
    settings = configs["sft"]
    manifest = prepare_sft_dataset(
        settings["dataset"],
        get_chat_tokenizer(**configs["tokenizer"]),
        settings["sequence_lengths"],
        output_dir=settings.get("data_dir"),
        restart=args.restart,
        source_dir=args.source_dir,
    )
    print(
        f"Prepared {manifest['splits']['train']['samples']:,} training conversations "
        f"and {manifest['splits']['val']['samples']:,} validation conversations"
    )


if __name__ == "__main__":
    main()
