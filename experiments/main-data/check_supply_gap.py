"""Verify the recorded supply gap still matches what its inputs derive.

The gap is the release's real constraint, so a stale copy of it is worse than
none: it would understate how far acquisition has to go. This rebuilds the
report from the pinned receipts and rejects any divergence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Importable from any working directory, so the Makefile does not have to cd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_supply_gap import build  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def validate(path: str | Path = ROOT / "experiments/main-data/supply-gap.json") -> dict:
    recorded = json.loads(Path(path).resolve().read_text())
    derived = build()
    if recorded != derived:
        raise ValueError(
            "supply-gap.json is stale; rerun experiments/main-data/build_supply_gap.py"
        )
    if recorded["totals"]["eligible_unique_tokens_established"] != 0:
        raise ValueError(
            "eligible tokens are recorded but no gate closure supports them; "
            "update source-readiness.json first"
        )
    return {
        "format": recorded["format"],
        "status": recorded["status"],
        "banks": len(recorded["banks"]),
        "retained_coverage_percent": recorded["totals"]["retained_coverage_percent"],
        "shortfall_tokens": recorded["totals"]["shortfall_tokens"],
        "zero_stock_banks": recorded["zero_stock_banks"]["ids"],
        "binding_bank": recorded["binding_constraint"]["bank"],
        "binding_percent_of_working_horizon": recorded["binding_constraint"][
            "percent_of_working_horizon"
        ],
        "training_admitted": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=ROOT / "experiments/main-data/supply-gap.json")
    args = parser.parse_args()
    print(json.dumps(validate(args.path), indent=2, sort_keys=True))
