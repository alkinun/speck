"""Run a hash-pinned bounded Stack v3 extraction successor."""

import argparse

from speck.stack_v3_expand import load_expansion_config, qualify_stack_v3_expansion


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Stack v3 expansion overlay JSON")
    parser.add_argument("--restart", action="store_true", help="replace incomplete derived output")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    expansion = load_expansion_config(args.config)
    report = qualify_stack_v3_expansion(expansion, restart=args.restart)
    print(
        f"Expanded bounded sample to "
        f"{sum(value['sampled_bytes'] for value in report['accepted_languages'].values()):,} bytes"
    )
    print(f"Non-authoritative result: {expansion['output_directory']}/report.json")


if __name__ == "__main__":
    main()
