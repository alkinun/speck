"""Apply a pinned fully redacted Gitleaks report to a source JSONL artifact."""

import argparse

from speck.gitleaks_filter import (
    apply_gitleaks_filter,
    load_gitleaks_filter_config,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Gitleaks exclusion plan JSON")
    parser.add_argument("--restart", action="store_true", help="replace incomplete derived output")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_gitleaks_filter_config(args.config)
    report = apply_gitleaks_filter(config, restart=args.restart)
    print(
        f"Excluded {report['counts'].get('records_rejected_gitleaks', 0):,} of "
        f"{report['counts']['records_seen']:,} records"
    )
    print(f"Non-authoritative result: {config['output_directory']}/report.json")


if __name__ == "__main__":
    main()
