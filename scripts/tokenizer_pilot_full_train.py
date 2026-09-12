"""Run one immutable, authorized tokenizer-pilot screen arm."""

import argparse
import json

from speck.tokenizer_pilot_full_train import run_authorized_training
from speck.tokenizer_pilot_orchestration import load_execution_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("execution")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = run_authorized_training(load_execution_record(args.execution), device=args.device)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
