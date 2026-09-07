"""Fetch a bounded, high-score Stack-Edu sample from pinned SWH metadata."""

import argparse

from speck.stack_edu import load_stack_edu_config, sample_stack_edu


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Stack-Edu sample plan JSON")
    parser.add_argument("--restart", action="store_true", help="replace incomplete derived output")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_stack_edu_config(args.config)
    report = sample_stack_edu(config, restart=args.restart)
    print(
        f"Sampled {report['counts']['records_sampled']:,} Stack-Edu files from "
        f"{report['repositories']:,} repositories"
    )
    print(f"Non-authoritative result: {config['output_directory']}/report.json")


if __name__ == "__main__":
    main()
