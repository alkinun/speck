"""Freeze a calibrated firewall construction config from prepared inputs."""

import argparse

from speck.firewall_inputs import freeze_calibrated_firewall_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("firewall_plan")
    parser.add_argument("prepared_manifest")
    parser.add_argument("destination")
    args = parser.parse_args()
    result = freeze_calibrated_firewall_config(
        args.firewall_plan, args.prepared_manifest, args.destination
    )
    print(f"Froze {len(result['categories'])} calibrated firewall categories.")


if __name__ == "__main__":
    main()
