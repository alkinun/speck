"""Compare local per-token log-probability artifacts under a frozen offline contract."""

import argparse
import json
from pathlib import Path

from speck.logprob_parity import compare_artifacts


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidates", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    try:
        report = compare_artifacts(args.plan, args.reference, args.candidates)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except (OSError, ValueError) as error:
        raise SystemExit(f"log-probability parity failed: {error}") from error
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
