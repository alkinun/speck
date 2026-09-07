"""Fetch a bounded MegaMath code tokenizer sample."""

import argparse

from speck.megamath_code import load_megamath_code_config, sample_megamath_code


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="MegaMath code sample plan JSON")
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    report = sample_megamath_code(load_megamath_code_config(args.config), restart=args.restart)
    observed = report["downstream_partition"]["observed"]
    print(
        f"Sampled {report['counts']['records_sampled']:,} files; "
        f"train={observed['train_bytes']:,}, eval={observed['eval_bytes']:,} bytes."
    )


if __name__ == "__main__":
    main()
