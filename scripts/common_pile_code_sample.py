"""Extract a bounded, high-score Common Pile Stack v2 educational sample."""

import argparse

from speck.common_pile_code import (
    load_common_pile_code_config,
    sample_common_pile_code,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Common Pile code sample plan JSON")
    parser.add_argument("--restart", action="store_true", help="replace incomplete derived output")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_common_pile_code_config(args.config)
    report = sample_common_pile_code(config, restart=args.restart)
    print(
        f"Sampled {report['counts']['records_sampled']:,} files and "
        f"{report['sampled_bytes']:,} content bytes"
    )
    print(f"Non-authoritative result: {config['output_directory']}/report.json")


if __name__ == "__main__":
    main()
