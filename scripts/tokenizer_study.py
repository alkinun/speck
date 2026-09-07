"""Run bounded exploratory tokenizer treatments and diagnostics."""

import argparse
import json
from pathlib import Path

from speck.tokenizer_study import evaluate_study, train_study_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("train", "evaluate"))
    parser.add_argument("spec")
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    result = train_study_model(spec) if args.command == "train" else evaluate_study(spec)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
