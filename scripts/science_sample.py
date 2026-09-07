"""Build one bounded science tokenizer source."""

import argparse

from speck.science_sample import load_science_sample_config, sample_science_source


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="science source plan JSON")
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    report = sample_science_source(load_science_sample_config(args.config), restart=args.restart)
    observed = report["downstream_partition"]["observed"]
    print(
        f"Sampled {report['counts']['records_sampled']:,} documents; "
        f"train={observed['train_bytes']:,}, eval={observed['eval_bytes']:,} bytes."
    )


if __name__ == "__main__":
    main()
