"""Build one bounded synthetic tokenizer source."""

import argparse

from speck.synthetic_sample import load_synthetic_sample_config, sample_synthetic_source


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="synthetic sample plan JSON")
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    report = sample_synthetic_source(
        load_synthetic_sample_config(args.config), restart=args.restart
    )
    print(f"Sampled {report['counts']['records_sampled']:,} records.")


if __name__ == "__main__":
    main()
