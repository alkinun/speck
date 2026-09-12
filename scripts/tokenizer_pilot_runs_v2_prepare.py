"""Materialize corrected, execution-blocked tokenizer-pilot screen records."""

import argparse
import json

from speck.tokenizer_pilot_runs_v2 import (
    load_corrected_materialization_plan,
    materialize_corrected_screen_runs,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan")
    args = parser.parse_args()
    result = materialize_corrected_screen_runs(load_corrected_materialization_plan(args.plan))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
