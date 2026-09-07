"""Build one bounded web tokenizer source."""

import argparse

from speck.web_sample import load_web_sample_config, sample_web_source


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="web sample plan JSON")
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    report = sample_web_source(load_web_sample_config(args.config), restart=args.restart)
    observed = report["downstream_partition"]["observed"]
    print(
        f"Sampled {report['counts']['records_sampled']:,} {report['source']['id']} documents: "
        f"{observed['train_bytes']:,} train bytes, {observed['eval_bytes']:,} eval bytes."
    )


if __name__ == "__main__":
    main()
