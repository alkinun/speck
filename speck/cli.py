"""Installed command interface; implementations are imported only when selected."""

import argparse
import importlib
import sys

COMMANDS = {
    "train": "speck.training.base",
    "sft": "speck.training.sft",
    "infer": "speck.evaluation.inference",
    "evaluate": "speck.evaluation.loss",
    "benchmark": "speck.training.benchmark",
    "export": "speck.export.transformers",
}


def main(argv=None):
    values = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(values)
    module = importlib.import_module(COMMANDS[args.command])
    original = sys.argv
    try:
        sys.argv = [f"speck {args.command}", *args.arguments]
        return module.main()
    finally:
        sys.argv = original


if __name__ == "__main__":
    main()
