"""Run or resume a frozen data rehearsal and optionally issue operations authority."""

import argparse
import json

from speck.data_rehearsal import (
    issue_operations_qualification,
    load_rehearsal_config,
    run_rehearsal,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--issue-operations-record")
    args = parser.parse_args()
    config = load_rehearsal_config(args.config)
    result = run_rehearsal(config, restart=args.restart)
    if args.issue_operations_record:
        result = issue_operations_qualification(
            f"{config['output_directory']}/manifest.json", args.issue_operations_record
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
