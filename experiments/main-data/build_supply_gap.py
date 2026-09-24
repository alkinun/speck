"""Derive the per-bank supply gap of the parent run from the plan and stock receipts.

Eligible tokens bound this release more than compute does. This regenerates `supply-gap.json`
from the parent's starting mixture and the retained-stock receipts, so the gap moves as
acquisition proceeds and is never typed into a document. It admits nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Which retained inventories can be candidates for each bank. This maps identity, not admission.
BANK_STOCK = {
    "selected_web": ("ultrafineweb_hq_distinct",),
    "independent_web": ("fineweb_edu",),
    "natural_code": ("retained_code", "acquired_code"),
    "natural_math": ("finemath_4plus",),
    "reference_science": ("pes2o_v3", "finewiki_en"),
    "refined_web": ("cosmopedia_v2",),
}

SOURCES = [
    "experiments/main-data/plan.json",
    "experiments/pilot/supply.json",
    "experiments/corpus-audit/data-readiness.json",
    "experiments/corpus-audit/stack-edu-acquisition.json",
    "experiments/main-data/source-readiness.json",
]


def _load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def build() -> dict:
    plan, supply, closeout, acquisition, readiness = (_load(path) for path in SOURCES)
    parent = plan["parent"]

    stock = {entry["source"]: entry["tokens"] for entry in supply["sources"]}
    stock["retained_code"] = supply["code"]["tokens_before_full_exclusion"]
    # Tranches fetched since, through the same screen and disjoint from the retained rows.
    stock["acquired_code"] = acquisition["tokens_before_full_exclusion"]
    # The HQ bank's measured candidate quantity is the part distinct from retained FineWeb-Edu.
    stock["ultrafineweb_hq_distinct"] = closeout["pretraining"]["hq"][
        "distinct_from_retained_fineweb_tokens"
    ]

    banks = []
    for item in parent["starting_mixture"]:
        share = item["weight_percent"] / 100
        target = round(parent["target_tokens"] * parent["preparation_factor"] * share)
        retained = sum(stock[name] for name in BANK_STOCK[item["id"]])
        banks.append(
            {
                "id": item["id"],
                "weight_percent": item["weight_percent"],
                "preparation_target_tokens": target,
                "retained_candidate_stock_tokens": retained,
                "retained_stock_sources": list(BANK_STOCK[item["id"]]),
                "coverage_percent": round(100 * retained / target, 2),
                "one_pass_exposure_cap_tokens": int(retained / share),
            }
        )

    target_total = sum(bank["preparation_target_tokens"] for bank in banks)
    retained_total = sum(bank["retained_candidate_stock_tokens"] for bank in banks)
    binding = min(banks, key=lambda bank: bank["one_pass_exposure_cap_tokens"])
    selected = [source for source in readiness["sources"] if source.get("selected")]

    return {
        "format": "speck_supply_gap",
        "format_version": 2,
        "status": "derived_from_receipts_no_admission",
        "generated_by": "experiments/main-data/build_supply_gap.py",
        "source_of_truth": SOURCES,
        "parent_target_tokens": parent["target_tokens"],
        "banks": banks,
        "totals": {
            "preparation_target_tokens": target_total,
            "retained_candidate_stock_tokens": retained_total,
            "coverage_percent": round(100 * retained_total / target_total, 2),
            "eligible_unique_tokens_established": 0,
        },
        "binding_bank": {
            "id": binding["id"],
            "one_pass_exposure_cap_tokens": binding["one_pass_exposure_cap_tokens"],
        },
        "open_gates_over_selected_sources": {
            gate: sum(1 for source in selected if source["gates"][gate] != "closed")
            for gate in ("source_use", "family_partition", "finite_supply")
        },
        "boundary": (
            "Retained stock is candidate supply with gates still open, an upper bound on what they "
            "could admit. Targets assume the parent's starting mixture, which the ladder will "
            "revise; ladder runs need far fewer tokens than the parent. No source is admitted."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "experiments/main-data/supply-gap.json"
    )
    args = parser.parse_args()
    report = build()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["totals"] | {"binding_bank": report["binding_bank"]}, indent=2))


if __name__ == "__main__":
    main()
