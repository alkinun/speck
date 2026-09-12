"""Run a two-step tokenizer-pilot checkpoint/replay qualification."""

import argparse
import json
from pathlib import Path

import torch

from speck.io import atomic_json
from speck.tokenizer_pilot_runtime import load_pilot_run_manifest
from speck.tokenizer_pilot_train import qualify_checkpoint_resume


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    result = qualify_checkpoint_resume(
        load_pilot_run_manifest(args.run),
        args.output_directory,
        device=args.device,
    )
    atomic_json(args.output_directory / "qualification.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
