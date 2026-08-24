"""Prepare packed assistant-masked SpeckChat1 data for instruction tuning."""

import argparse

from speck.chat import get_chat_tokenizer
from speck.config import load_experiment
from speck.sft import prepare_sft_dataset

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "experiment",
    nargs="?",
    default="experiments/Speck1-140M",
    help="experiment directory (default: %(default)s)",
)
parser.add_argument("--restart", action="store_true", help="replace an incomplete staged build")
args = parser.parse_args()

configs = load_experiment(args.experiment, "tokenizer", "sft")
settings = configs["sft"]
manifest = prepare_sft_dataset(
    settings["dataset"],
    get_chat_tokenizer(**configs["tokenizer"]),
    settings["sequence_lengths"],
    output_dir=settings.get("data_dir"),
    restart=args.restart,
)
print(
    f"prepared {manifest['splits']['train']['samples']:,} training conversations "
    f"and {manifest['splits']['val']['samples']:,} validation conversations"
)
