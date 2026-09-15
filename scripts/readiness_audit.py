"""Audit the selected pre-access research contracts; a passing audit is not launch authority."""

import argparse
import json
from pathlib import Path

from speck.provenance.io import durable_json
from speck.provenance.readiness import audit_readiness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError("preserve the previous audit and choose a new output")
    result = audit_readiness()
    if args.output is not None:
        durable_json(args.output, result)
    print(
        json.dumps(
            {key: result[key] for key in ("status", "training_authority", "pending")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
