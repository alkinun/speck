"""Derive the per-bank supply gap from the plan and retained-stock receipts.

Compute is not what bounds this release; eligible tokens are. This regenerates
`supply-gap.json` from the working mixture and the retained-stock receipts so the
gap is a computed quantity that moves as acquisition proceeds, never a number
typed into a document. It admits nothing and acquires nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Which retained inventories can even be candidates for each declared bank.
# Retained stock is not eligible supply; this maps identity, not admission.
BANK_STOCK = {
    "selected_web": ("ultrafineweb_hq_distinct",),
    "independent_web": ("fineweb_edu",),
    "natural_code": ("retained_code",),
    "natural_math": ("finemath_4plus",),
    "reference_science": ("pes2o_v3", "finewiki_en"),
    "refined_web": ("cosmopedia_v2",),
}


def _load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def build() -> dict:
    plan = _load("experiments/main-data/plan.json")
    supply = _load("experiments/pilot/supply.json")
    readiness = _load("experiments/main-data/source-readiness.json")

    stock = {entry["source"]: entry["tokens"] for entry in supply["sources"]}
    stock["retained_code"] = supply["code"]["tokens_before_full_exclusion"]
    # The HQ bank's only measured candidate quantity is the portion distinct from
    # retained FineWeb-Edu, recorded as a one-pass constraint numerator.
    constraints = {
        item["source"]: item for item in readiness["horizon_accounting"]["one_pass_constraints"]
    }
    stock["ultrafineweb_hq_distinct"] = constraints[
        "ultrafineweb_hq_distinct_from_retained_fineweb_edu"
    ]["numerator_tokens"]

    # Gate counts are only meaningful over sources the mixture actually selects. Counting a
    # dropped candidate as "open" would overstate the remaining work forever, since a source
    # nobody intends to use is never going to have its gates closed.
    selected = [source for source in readiness["sources"] if source.get("selected")]
    dropped = [source for source in readiness["sources"] if not source.get("selected")]
    gate_status = {source["id"]: source["gates"] for source in selected}

    banks = []
    for item in plan["main_pretraining"]["mixture"]:
        names = BANK_STOCK[item["id"]]
        retained = sum(stock[name] for name in names)
        target = item["eligible_unique_token_preparation_target"]
        banks.append(
            {
                "id": item["id"],
                "role": item["role"],
                "weight_percent": item["weight_percent"],
                "exposure_tokens": item["exposure_tokens"],
                "eligible_unique_token_preparation_target": target,
                "retained_candidate_stock_tokens": retained,
                "retained_stock_sources": list(names),
                "eligible_unique_tokens": 0,
                "shortfall_tokens": target - retained,
                "retained_coverage_percent": round(100.0 * retained / target, 3),
                "maximum_one_pass_exposure_from_retained_stock": (
                    int(retained // (item["weight_percent"] / 100.0))
                    if item["weight_percent"]
                    else None
                ),
            }
        )

    target_total = plan["main_pretraining"]["target_unique_eligible_tokens"]
    retained_total = sum(bank["retained_candidate_stock_tokens"] for bank in banks)
    zero_stock = [bank["id"] for bank in banks if not bank["retained_candidate_stock_tokens"]]
    # Banks with no stock at all cap a one-pass run at zero, which is true but
    # uninformative. Report them separately and take the binding constraint among
    # banks that have something, so the number says which acquisition matters most.
    binding = min(
        (
            bank
            for bank in banks
            if bank["weight_percent"] and bank["retained_candidate_stock_tokens"]
        ),
        key=lambda bank: bank["maximum_one_pass_exposure_from_retained_stock"],
    )

    return {
        "format": "speck_supply_gap",
        "format_version": 1,
        "status": "derived_from_receipts_no_admission",
        "purpose": (
            "State the distance between the declared mixture and measured retained stock, so "
            "acquisition is planned against a number rather than an impression."
        ),
        "generated_by": "experiments/main-data/build_supply_gap.py",
        "source_of_truth": [
            "experiments/main-data/plan.json",
            "experiments/pilot/supply.json",
            "experiments/main-data/source-readiness.json",
        ],
        "banks": banks,
        "totals": {
            "eligible_unique_token_preparation_target": target_total,
            "retained_candidate_stock_tokens": retained_total,
            "eligible_unique_tokens_established": 0,
            "shortfall_tokens": target_total - retained_total,
            "retained_coverage_percent": round(100.0 * retained_total / target_total, 3),
            "acquisition_multiple_required": round(target_total / retained_total, 2),
        },
        "zero_stock_banks": {
            "ids": zero_stock,
            "combined_preparation_target_tokens": sum(
                bank["eligible_unique_token_preparation_target"]
                for bank in banks
                if bank["id"] in zero_stock
            ),
            "meaning": (
                "These banks have no retained candidate stock of any kind, so at the declared "
                "weights a one-pass run is capped at zero tokens until they are supplied or the "
                "mixture is re-frozen without them. Treat that as a mixture decision, not an "
                "acquisition detail."
            )
            if zero_stock
            else (
                "Every declared bank has some retained candidate stock, so no bank caps a "
                "one-pass run at zero. The 2026-09-22 re-freeze removed the two that did. This "
                "says nothing about eligibility: zero eligible tokens are still established."
            ),
        },
        "binding_constraint": {
            "bank": binding["id"],
            "scope": "binding among banks that have any retained stock",
            "maximum_total_exposure_tokens": binding[
                "maximum_one_pass_exposure_from_retained_stock"
            ],
            "working_horizon_tokens": plan["main_pretraining"]["target_tokens"],
            "percent_of_working_horizon": round(
                100.0
                * binding["maximum_one_pass_exposure_from_retained_stock"]
                / plan["main_pretraining"]["target_tokens"],
                3,
            ),
            "meaning": (
                "At its declared share, this bank's retained stock alone caps a one-pass run at "
                "this exposure. Every other bank could be complete and the horizon would still "
                "bind here."
            ),
        },
        "gate_summary": {
            "sources_tracked": len(gate_status),
            "sources_not_selected": [source["id"] for source in dropped],
            "source_use_open": sum(
                1 for gates in gate_status.values() if gates["source_use"] != "closed"
            ),
            "family_partition_open": sum(
                1 for gates in gate_status.values() if gates["family_partition"] != "closed"
            ),
            "finite_supply_open": sum(
                1 for gates in gate_status.values() if gates["finite_supply"] != "closed"
            ),
        },
        "boundary": (
            "Retained stock is not eligible supply: zero eligible tokens are established, so "
            "every figure here is an upper bound on what the gates could admit, not a promise "
            "of usable data. Shortfalls assume the declared weights; re-freezing the mixture "
            "changes them. No source is admitted and no acquisition is authorized."
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
    print(json.dumps(report["totals"] | report["binding_constraint"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
