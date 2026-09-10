"""Build the real firewall under the accepted production-calibration fallback."""

import argparse

from speck.data_firewall_calibrated import (
    construct_calibrated_firewall,
    load_calibrated_firewall_config,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    result = construct_calibrated_firewall(
        load_calibrated_firewall_config(args.config), restart=args.restart
    )
    print(f"Built {len(result['categories'])} production firewall categories.")


if __name__ == "__main__":
    main()
