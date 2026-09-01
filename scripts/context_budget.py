"""Report context-dependent training FLOPs and model-state memory from a real config."""

import argparse
import json
from pathlib import Path

import torch

from speck.budget import estimate_context_budget
from speck.config import load_experiment
from speck.long_context import parse_lengths
from speck.model import build_model


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--lengths", type=parse_lengths, default=None)
    parser.add_argument("--effective-tflops", type=float, default=400.0)
    parser.add_argument("--h100-hours", type=float, default=10_000.0)
    parser.add_argument("--weight-bits", type=float, default=4.0)
    parser.add_argument(
        "--kv-cache-dtype",
        choices=("bfloat16", "float16", "float32", "int8"),
        default="int8",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def run(args):
    configs = load_experiment(args.experiment, "long_context", "model")
    lengths = args.lengths or tuple(configs["long_context"]["lengths"])
    with torch.device("meta"):
        model = build_model(configs["model"], vocab_size=32_000)
    report = estimate_context_budget(
        model,
        lengths,
        effective_tflops=args.effective_tflops,
        h100_hours=args.h100_hours,
        weight_bits=args.weight_bits,
        kv_cache_dtype=args.kv_cache_dtype,
    )
    report["format"] = "speck_context_budget"
    report["format_version"] = 1
    report["experiment"] = str(args.experiment.expanduser().resolve())
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return report


def main(argv=None):
    run(arguments(argv))


if __name__ == "__main__":
    main()
