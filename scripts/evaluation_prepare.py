"""Verify and partition the pilot's pinned public evaluation inputs."""

import argparse

from speck.evaluation.protocol import prepare_protocol


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = prepare_protocol(args.protocol, args.output)
    for name, splits in result["partitions"].items():
        print(f"{name}: {len(splits['development'])} development, {len(splits['final'])} final")


if __name__ == "__main__":
    main()
