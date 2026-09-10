"""Run or resume the frozen 2B production-data calibration."""

import argparse
import json

from speck.production_calibration import (
    load_production_calibration_plan,
    run_production_calibration,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan")
    args = parser.parse_args()
    result = run_production_calibration(load_production_calibration_plan(args.plan))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
