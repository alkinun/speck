"""Run a limited official ArithMark evaluation against a local model export."""

import argparse
import sys
from pathlib import Path

from scripts import open_slm_eval


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=("arithmark-2", "arithmark-3"))
    parser.add_argument("model", type=Path)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=open_slm_eval.DEFAULT_CONFIG)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    return args


def limited_loader(module, name, limit):
    original = getattr(module, name)

    def load(path):
        return original(path)[:limit]

    setattr(module, name, load)


def run_arithmark_2(config, files, model, limit, output_dir):
    official = open_slm_eval._load_module(files["runner"], "limited_arithmark_2")
    limited_loader(official, "load_arithmark_2", limit)
    open_slm_eval._disable_arithmark_2_cache(official)
    official.CACHE_DIR = str(output_dir)
    sys.argv = [
        str(files["runner"]),
        "--model",
        str(model),
        "--batch-size",
        str(config["arithmark_2"]["batch_size"]),
        "--data-path",
        str(files["data"]),
    ]
    official.main()


def run_arithmark_3(config, files, model, limit, output_dir, device):
    official = open_slm_eval._load_module(files["runner"], "limited_arithmark_3")
    limited_loader(official, "load_arithmark_3", limit)
    benchmark = config["arithmark_3"]
    sys.argv = [
        str(files["runner"]),
        "--model",
        str(model),
        "--batch-size",
        str(benchmark["batch_size"]),
        "--max-context",
        str(benchmark["max_context"]),
        "--data-path",
        str(files["data"]),
        "--device",
        device,
        "--dtype",
        benchmark["dtype"],
        "--primary-metric",
        benchmark["primary_metric"],
        "--results-dir",
        str(output_dir),
    ]
    official.main()


def main():
    args = arguments()
    config = open_slm_eval._load_config(args.config)
    key = args.benchmark.replace("-", "_")
    files = open_slm_eval._download_benchmark_files(config)[key]
    model = args.model.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    original_argv = sys.argv[:]
    try:
        if args.benchmark == "arithmark-2":
            run_arithmark_2(config, files, model, args.limit, output_dir)
        else:
            run_arithmark_3(config, files, model, args.limit, output_dir, args.device)
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
