"""Exclude Gitleaks-affected records from a bounded text source."""

import argparse

from speck.text_gitleaks_filter import (
    apply_text_gitleaks_filter,
    load_text_gitleaks_config,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="text Gitleaks exclusion plan JSON")
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    report = apply_text_gitleaks_filter(
        load_text_gitleaks_config(args.config), restart=args.restart
    )
    print(
        f"Excluded {report['counts'].get('records_rejected_gitleaks', 0):,} of "
        f"{report['counts']['records_seen']:,} records."
    )


if __name__ == "__main__":
    main()
