"""Download and profile only the pinned Stack v3 shards in a qualification plan."""

import argparse

from speck.stack_v3 import load_stack_v3_config, qualify_stack_v3


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Stack v3 qualification JSON")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="discard and rebuild incomplete derived outputs; downloaded shards are retained",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_stack_v3_config(args.config)
    report = qualify_stack_v3(config, restart=args.restart)
    print(
        f"Profiled {report['counts']['repositories_seen']:,} repositories and "
        f"{report['counts']['files_seen']:,} files"
    )
    print(f"Bounded non-authoritative result: {config['output']['directory']}/report.json")


if __name__ == "__main__":
    main()
