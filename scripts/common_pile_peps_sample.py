"""Build the bounded Common Pile Python PEP tokenizer source."""

import argparse

from speck.common_pile_peps import load_peps_config, sample_common_pile_peps


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="PEP sample plan JSON")
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    report = sample_common_pile_peps(load_peps_config(args.config), restart=args.restart)
    observed = report["downstream_partition"]["observed"]
    print(
        f"Sampled {report['counts']['records_sampled']:,} PEPs: "
        f"{observed['train_bytes']:,} train bytes, {observed['eval_bytes']:,} eval bytes."
    )


if __name__ == "__main__":
    main()
