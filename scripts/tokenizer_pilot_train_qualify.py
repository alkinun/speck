"""Run a two-step tokenizer-pilot checkpoint/replay qualification."""

import argparse
import json
import traceback
from pathlib import Path

import torch

from speck.io import atomic_json
from speck.tokenizer_pilot_runtime import load_pilot_run_manifest
from speck.tokenizer_pilot_train import load_resume_policy, qualify_checkpoint_resume


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume-policy", type=Path, default=None)
    args = parser.parse_args()
    try:
        result = qualify_checkpoint_resume(
            load_pilot_run_manifest(args.run),
            args.output_directory,
            device=args.device,
            resume_policy=(
                load_resume_policy(args.resume_policy) if args.resume_policy is not None else None
            ),
        )
    except Exception as error:
        if args.output_directory.is_dir():
            atomic_json(
                args.output_directory / "failure.json",
                {
                    "format": "speck_tokenizer_pilot_training_qualification_failure",
                    "format_version": 1,
                    "status": "failed_no_screen_authority",
                    "run": str(Path(args.run).resolve()),
                    "device": args.device,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "scientific_run": False,
                    "screen_execution_authority": False,
                    "D5_opening_authority": False,
                },
            )
        raise
    atomic_json(args.output_directory / "qualification.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
