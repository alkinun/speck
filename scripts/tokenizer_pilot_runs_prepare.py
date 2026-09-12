"""Materialize the three immutable tokenizer-pilot screen run manifests."""

import argparse
import json

from speck.tokenizer_pilot_runs import load_run_materialization_plan, materialize_screen_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan")
    args = parser.parse_args()
    result = materialize_screen_runs(load_run_materialization_plan(args.plan))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
