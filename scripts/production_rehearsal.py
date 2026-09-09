"""Run one stage of the frozen flagship production data rehearsal."""

import argparse
import json

from speck.production_rehearsal import (
    load_production_rehearsal_plan,
    run_production_rehearsal_stage,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan")
    parser.add_argument(
        "stage",
        choices=(
            "source_identity",
            "acquisition",
            "global_dedup",
            "packing",
            "resume_cleanup",
            "firewall_disjointness",
        ),
    )
    parser.add_argument("result")
    args = parser.parse_args()
    result = run_production_rehearsal_stage(
        load_production_rehearsal_plan(args.plan), args.stage, args.result
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
