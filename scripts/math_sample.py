"""Build one bounded math tokenizer source from an immutable shard."""

import argparse

from speck.math_sample import load_math_sample_config, sample_math_source


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="math sample plan JSON")
    parser.add_argument("--restart", action="store_true", help="replace incomplete staging output")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    report = sample_math_source(load_math_sample_config(args.config), restart=args.restart)
    observed = report["downstream_partition"]["observed"]
    print(
        f"Sampled {report['counts']['records_sampled']:,} records; "
        f"train={observed['train_bytes']:,} bytes, eval={observed['eval_bytes']:,} bytes."
    )


if __name__ == "__main__":
    main()
