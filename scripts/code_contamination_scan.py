"""Scan a refined code corpus against pinned code-evaluation payloads."""

import argparse

from speck.code_contamination import load_contamination_config, scan_code_contamination


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="code contamination plan JSON")
    parser.add_argument("--restart", action="store_true", help="replace incomplete derived output")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_contamination_config(args.config)
    report = scan_code_contamination(config, restart=args.restart)
    print(
        f"Scanned {report['counts']['records_seen']:,} code files; removed "
        f"{report['counts'].get('records_removed_critical', 0):,} critical matches"
    )
    print(f"Non-authoritative result: {config['output_directory']}/report.json")


if __name__ == "__main__":
    main()
