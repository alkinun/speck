"""Run a frozen code-benchmark scan for a generic code-source successor."""

import argparse

from speck.code_contamination_successor import (
    load_successor_config,
    scan_code_contamination_successor,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="successor contamination plan JSON")
    parser.add_argument("--restart", action="store_true", help="replace incomplete staging output")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    report = scan_code_contamination_successor(
        load_successor_config(args.config), restart=args.restart
    )
    print(
        f"Retained {report['counts']['records_retained']:,} records; "
        f"removed {report['counts'].get('records_removed_critical', 0):,} critical records."
    )


if __name__ == "__main__":
    main()
