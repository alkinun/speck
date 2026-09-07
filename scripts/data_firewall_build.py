"""Construct a frozen flagship data firewall."""

import argparse

from speck.data_firewall import construct_firewall, load_firewall_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    report = construct_firewall(load_firewall_config(args.config), restart=args.restart)
    print(f"Built {len(report['categories'])} category firewall partitions.")


if __name__ == "__main__":
    main()
