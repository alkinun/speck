"""Validate the source-readiness matrix without authorizing data or training.

Run from the repository root::

    python experiments/main-data/check_source_readiness.py experiments/main-data/source-readiness.json

The check verifies the matrix's local receipt identities and non-admission boundary. It does not
acquire sources, execute corpus content, or infer rights decisions.
"""

import argparse
import json
from pathlib import Path

from speck.provenance.io import check_reference

ROOT = Path(__file__).resolve().parents[2]
CLOSEOUT = "experiments/corpus-audit/data-readiness.json"
ACCEPTANCE = "experiments/main-data/source-rights-acceptance.json"
CONTRACT = "experiments/main-data/data-design-contract.json"


def validate(matrix_path):
    matrix_path = Path(matrix_path).resolve()
    matrix = json.loads(matrix_path.read_text())
    closeout = json.loads((ROOT / CLOSEOUT).read_text())
    if matrix.get("format") != "speck_data_study_source_readiness":
        raise ValueError("unsupported source-readiness format")
    if matrix.get("format_version") != 1:
        raise ValueError("unsupported source-readiness version")
    if matrix.get("status") != "candidate_evidence_only_no_training_admission":
        raise ValueError("source-readiness matrix must remain evidence-only")

    receipts = matrix["source_of_truth"]
    for entry in receipts.values():
        check_reference(entry)

    sources = matrix["sources"]
    ids = [source["id"] for source in sources]
    if len(ids) != len(set(ids)):
        raise ValueError("source IDs must be unique")
    if not sources or any(source.get("admitted") is not False for source in sources):
        raise ValueError("every source must remain explicitly non-admitted")
    acceptance = json.loads((ROOT / ACCEPTANCE).read_text())
    if {source["id"] for source in sources if source["selected"]} != set(
        acceptance["approved_source_ids"]
    ):
        raise ValueError("selected sources differ from the signed source-use acceptance")
    vocabulary = matrix["manifest_field_vocabulary"]
    required = json.loads((ROOT / CONTRACT).read_text())["manifest_schema"]["required_fields"]
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
        # No manifest is complete until finite eligible supply exists, so no field may be bound.
        fields = source["manifest_fields"]
        if list(fields) != required or not set(fields.values()) <= vocabulary.keys():
            raise ValueError(f"manifest fields drift from the shared contract: {source['id']}")
        eligible = source.get("inventory", {}).get("eligible_unique_tokens_established", 0)
        if type(eligible) is not int or eligible != 0:
            raise ValueError(f"source cannot claim eligible tokens: {source['id']}")
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
        # A retained count that cites the closeout must equal it. The peS2o entry once kept
        # superseded stock-v1 counts, half the v2 stock the closeout and supply gap both use.
        for evidence in source.get("evidence", []):
            if evidence.get("path") != CLOSEOUT or "field" not in evidence:
                continue
            cited = closeout
            for key in evidence["field"].split("."):
                cited = cited[key]
            if isinstance(cited, dict) and {"documents", "tokens"} <= cited.keys():
                recorded = (inventory.get("documents"), inventory.get("retained_tokens"))
                if recorded != (cited["documents"], cited["tokens"]):
                    raise ValueError(f"retained inventory drifts from the closeout: {source['id']}")

    boundary = matrix.get("launch_boundary", "")
    if not boundary.startswith("This matrix records evidence and blockers only"):
        raise ValueError("matrix boundary must keep launch authority outside the artifact")
    return {
        "format": matrix["format"],
        "status": matrix["status"],
        "sources": len(sources),
        "admitted_sources": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.matrix), indent=2, sort_keys=True))
