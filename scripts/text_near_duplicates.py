"""Run exact and verified MinHash overlap analysis for bounded text sources."""

import argparse

from speck.text_near_duplicates import (
    analyze_text_cross_source_duplicates,
    load_text_duplicate_config,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="text duplicate analysis plan JSON")
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    report = analyze_text_cross_source_duplicates(
        load_text_duplicate_config(args.config), restart=args.restart
    )
    print(
        f"Cross-source duplicates: exact={len(report['exact_cross_source_matches']):,}, "
        f"near={len(report['near_cross_source_matches']):,}."
    )


if __name__ == "__main__":
    main()
