"""Run frozen exact and MinHash/LSH overlap analysis across code sources."""

import argparse

from speck.code_near_duplicates import (
    analyze_cross_source_duplicates,
    load_duplicate_config,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="cross-source duplicate plan JSON")
    parser.add_argument("--restart", action="store_true", help="replace incomplete derived output")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_duplicate_config(args.config)
    report = analyze_cross_source_duplicates(config, restart=args.restart)
    print(
        f"Cross-source duplicates: exact={len(report['exact_cross_source_matches']):,}, "
        f"near={len(report['near_cross_source_matches']):,}"
    )
    print(f"Non-authoritative result: {config['output_directory']}/report.json")


if __name__ == "__main__":
    main()
