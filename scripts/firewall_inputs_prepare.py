"""Prepare deterministic globally deduplicated inputs for the real firewall."""

import argparse
import json

from speck.firewall_inputs import load_firewall_input_plan, prepare_firewall_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan")
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    result = prepare_firewall_inputs(load_firewall_input_plan(args.plan), restart=args.restart)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
