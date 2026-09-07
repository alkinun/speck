"""Run a frozen benchmark-contamination scan for bounded text sources."""

import argparse

from speck.text_contamination import load_text_contamination_config, scan_text_contamination


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="text contamination plan JSON")
    parser.add_argument("--restart", action="store_true", help="replace incomplete staging output")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    report = scan_text_contamination(
        load_text_contamination_config(args.config), restart=args.restart
    )
    removed = sum(
        source["counts"].get("records_removed_critical", 0) for source in report["sources"]
    )
    retained = sum(source["counts"]["records_retained"] for source in report["sources"])
    print(f"Retained {retained:,} records; removed {removed:,} critical records.")


if __name__ == "__main__":
    main()
