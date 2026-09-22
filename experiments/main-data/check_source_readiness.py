"""Validate the source-readiness matrix without authorizing data or training.

Run from the repository root::

    python experiments/main-data/check_source_readiness.py experiments/main-data/source-readiness.json

The check verifies the matrix's local receipt identities and non-admission boundary. It does not
acquire sources, execute corpus content, infer rights decisions or select a study arm.
"""

import argparse
import json
from pathlib import Path

from speck.provenance.io import file_sha256

ROOT = Path(__file__).resolve().parents[2]


def _artifact(path, expected):
    resolved = (ROOT / path).resolve() if not Path(path).is_absolute() else Path(path)
    if file_sha256(resolved) != expected:
        raise ValueError(f"artifact checksum mismatch: {resolved}")
    return resolved


def validate(matrix_path):
    matrix_path = Path(matrix_path).resolve()
    matrix = json.loads(matrix_path.read_text())
    if matrix.get("format") != "speck_data_study_source_readiness":
        raise ValueError("unsupported source-readiness format")
    if matrix.get("format_version") != 1:
        raise ValueError("unsupported source-readiness version")
    if matrix.get("status") != "candidate_evidence_only_no_training_admission":
        raise ValueError("source-readiness matrix must remain evidence-only")

    receipts = matrix["source_of_truth"]
    for entry in receipts.values():
        _artifact(entry["path"], entry["sha256"])

    sources = matrix["sources"]
    ids = [source["id"] for source in sources]
    if len(ids) != len(set(ids)):
        raise ValueError("source IDs must be unique")
    if not sources or any(source.get("admitted") is not False for source in sources):
        raise ValueError("every source must remain explicitly non-admitted")
    for source in sources:
        if not source.get("evidence") or not source.get("blockers"):
            raise ValueError(f"source lacks evidence or blockers: {source['id']}")
        if set(source["gates"]) != {
            "source_use",
            "family_partition",
            "correctness",
            "finite_supply",
            "runtime",
        }:
            raise ValueError(f"incomplete gate set: {source['id']}")
        if source["id"] in {
            "finemath_4plus",
            "ultradata_math_l2_preview",
            "nemotron_cc_math_4plus",
        }:
            evidence = source.get("qualification_evidence", {})
            if evidence.get("training_admitted") is not False:
                raise ValueError(f"math evidence must remain non-admitted: {source['id']}")
            if (
                source["id"] == "finemath_4plus"
                and evidence.get("retained_stock", {}).get("eligible_tokens_established") != 0
            ):
                raise ValueError("FineMath retained stock must not claim eligible tokens")
            if (
                source["id"] == "ultradata_math_l2_preview"
                and evidence.get("retained_stock", {}).get("host_missing_documents") != 169058
            ):
                raise ValueError("UltraData-Math L2 host-lineage count changed")
            if (
                source["id"] == "nemotron_cc_math_4plus"
                and evidence.get("decision")
                != "do_not_use_until_whole_objects_are acquired and independently qualified"
            ):
                raise ValueError("Nemotron availability boundary changed")
        if source["id"] in {"stack_edu", "stack_v3", "checked_code"}:
            evidence = source.get("qualification_evidence", {})
            if evidence.get("training_admitted") is not False:
                raise ValueError(f"code evidence must remain non-admitted: {source['id']}")
            if (
                source["id"] == "stack_edu"
                and evidence.get("retained_stock", {}).get("eligible_tokens_established") != 0
            ):
                raise ValueError("Stack-Edu retained stock must not claim eligible tokens")
            if source["id"] == "checked_code" and evidence.get("verified_origin_join") is not False:
                raise ValueError("checked-code route must keep its unresolved origin join")
        if source["id"] == "ultrafineweb_hq":
            comparison = source.get("comparison_evidence", {})
            if (
                comparison.get("eligible_tokens_established") != 0
                or comparison.get("training_admitted") is not False
            ):
                raise ValueError("HQ comparison must remain non-admitted")
            if (
                comparison.get("filter_validation_panel", {}).get("production_filter_changed")
                is not False
            ):
                raise ValueError("HQ filter validation must not silently change production filters")
            if (
                comparison.get("decision")
                != "retain_control_and_candidate_for_qualification; adopt_no_score_cutoff_or_repair_rule"
            ):
                raise ValueError("HQ comparison decision boundary changed")
        inventory = source.get("inventory", {})
        for key, value in inventory.items():
            if isinstance(value, int) and value < 0:
                raise ValueError(f"negative inventory count: {source['id']}.{key}")

    horizon = matrix["horizon_accounting"]
    target = horizon["working_target"]
    if target["minimum_unique_preparation_tokens"] != int(
        target["total_exposure_tokens"] * target["preparation_factor"]
    ):
        raise ValueError("working-horizon preparation arithmetic is inconsistent")
    # Each bound must name the bank whose weight it divides by, and that weight must be the one
    # the plan actually declares. Checking only the internal arithmetic is not enough: after the
    # 2026-09-22 re-freeze the code bound stayed self-consistent at a 30% share the mixture no
    # longer had, and understated the binding constraint by 17% while passing every check.
    mixture = json.loads((ROOT / "experiments/main-data/plan.json").read_text())
    weights = {
        item["id"]: item["weight_percent"] for item in mixture["main_pretraining"]["mixture"]
    }
    for bound in horizon["one_pass_constraints"]:
        bank = bound.get("bank")
        if bank not in weights:
            raise ValueError(f"one-pass bound names no declared bank: {bound['source']}")
        if bound["declared_share_percent"] != weights[bank]:
            raise ValueError(
                f"one-pass bound share drifts from the mixture: {bound['source']} declares "
                f"{bound['declared_share_percent']}%, bank {bank} carries {weights[bank]}%"
            )
        expected = (bound["numerator_tokens"] * 100) // bound["declared_share_percent"]
        if bound["maximum_total_exposure_tokens_before_exclusions"] != int(expected):
            raise ValueError(f"one-pass bound arithmetic is inconsistent: {bound['source']}")
    if horizon["observed_union"]["eligible_tokens_established"] != 0:
        raise ValueError("retained inventory must not claim eligible tokens")

    arms = matrix["arm_readiness"]
    expected_arms = {"baseline", "code_bank_candidate", "web_bank_candidate"}
    if set(arms) != expected_arms or any(
        value.get("status") != "blocked" for value in arms.values()
    ):
        raise ValueError("every packet arm must remain blocked")
    boundary = matrix.get("launch_boundary", "")
    if not boundary.startswith("This matrix records evidence and blockers only"):
        raise ValueError("matrix boundary must keep launch authority outside the artifact")
    return {
        "format": matrix["format"],
        "status": matrix["status"],
        "sources": len(sources),
        "admitted_sources": 0,
        "blocked_arms": len(arms),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.matrix), indent=2, sort_keys=True))
