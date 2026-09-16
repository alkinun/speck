"""Reconcile selected-source receipts and expose base-training demand without inferring eligibility."""

import json
from fractions import Fraction
from pathlib import Path

from speck.provenance.io import file_sha256
from speck.provenance.repository import repository_root

SOURCE_IDS = {
    "FineWeb-Edu": "fineweb_edu",
    "restricted Stack v3 under admitted source/language policy": "stack_v3_train_permissive",
    "FineMath-4+": "finemath_4plus",
    "Cosmopedia v2": "cosmopedia_v2",
    "peS2o v3": "pes2o_v3",
    "FineWiki": "finewiki_en",
}
CATEGORIES = ("web", "code", "math", "synthetic", "science", "reference")


def analyze(manifest_path, root=None):
    root = Path(root or repository_root()).resolve()
    manifest_path = Path(manifest_path).resolve()
    spec = json.loads(manifest_path.read_text())
    if spec.get("format") != "speck_base_supply_inputs" or spec.get("format_version") != 1:
        raise ValueError("unsupported base supply inputs")
    identities = []

    def load(binding):
        path = root / binding["path"]
        if file_sha256(path) != binding["sha256"]:
            raise ValueError("base supply input identity changed")
        identities.append(binding)
        return json.loads(path.read_text())

    data, model, decision = (load(spec[key]) for key in ("data_plan", "model_plan", "tokenizer"))
    if decision["status"] != "tokenizer_selected_and_frozen":
        raise ValueError("supply requires frozen tokenizer")
    base = data["base"]
    percentages = base["category_percent"]
    if (
        set(percentages) != set(CATEGORIES)
        or any(type(x) is not int or not 0 <= x <= 100 for x in percentages.values())
        or sum(percentages.values()) != 100
        or set(base["initial_sources"]) != set(CATEGORIES)
    ):
        raise ValueError("invalid selected category allocation")
    measured = {}
    for entry in spec["stocks"]:
        category = entry["category"]
        if category not in percentages or category in measured:
            raise ValueError(
                "duplicate or unknown stock category; overlapping banks cannot be added"
            )
        source = SOURCE_IDS[base["initial_sources"][category]]
        text = load(entry["text_result"])
        cache = load(entry["token_result"])
        cache_manifest = load(cache["manifest"])
        plan = cache_manifest["plan"]
        reopened = cache.get("completed_reopen_pass") is True
        if cache.get("completed_reopen_pass") is None and entry.get("reopen_receipt"):
            verification = load(entry["reopen_receipt"])
            if verification.get("format") != "speck_preparation_recovery_launch":
                raise ValueError("unsupported separate cache verification receipt")
            matching = [
                row
                for row in verification["completed_token_stock_post_outage_verification"]
                if row["source_id"] == source
            ]
            reopened = len(matching) == 1 and matching[0] == {
                "source_id": source,
                "documents": cache["documents"],
                "tokens": cache["tokens"],
                "manifest_sha256": cache["manifest"]["sha256"],
                "payload_hashes_and_index_coverage": "pass",
            }
        if (
            plan["source_id"] != source
            or plan["category"] != category
            or plan["stock_result"]["sha256"] != entry["text_result"]["sha256"]
            or plan["tokenizer_decision"]["sha256"] != spec["tokenizer"]["sha256"]
            or plan["tokenizer"]["sha256"] != decision["tokenizer_fingerprint"]
            or text["reference_capacity"]["tokenizer"]["sha256"]
            != decision["tokenizer_fingerprint"]
            or plan["parent_manifest"] != text["analysis"]["parent_manifest"]
        ):
            raise ValueError("stock source, text, exclusion or tokenizer lineage differs")
        if (
            not reopened
            or cache_manifest.get("status") != "complete_document_token_cache_not_training_view"
            or cache_manifest.get("training_authority") is not False
            or text.get("training_authority") is not False
            or text["analysis"]["exact_reference_overlap"] != 0
            or text["analysis"]["reference_outputs_preserved"] is not True
        ):
            raise ValueError("stock preparation/reopen qualification is incomplete")
        tokens = text["reference_capacity"]["tokens"]
        documents = text["reference_capacity"]["documents"]
        if (
            type(tokens) is not int
            or tokens <= 0
            or cache["tokens"] != tokens
            or cache_manifest["token_count"] != tokens
            or plan["expected_tokens"] != tokens
            or cache["documents"] != documents
            or cache_manifest["document_count"] != documents
            or plan["expected_documents"] != documents
        ):
            raise ValueError("text and cache counts differ")
        measured[category] = {
            "source_specific_stock_tokens": tokens,
            "documents": documents,
            "text_result": entry["text_result"],
            "token_result": entry["token_result"],
            "token_manifest": cache["manifest"],
        }
    scenarios = []
    for name, horizon in (
        ("default", model["flagship"]["base_tokens"]),
        ("stretch", model["flagship"]["stretch_tokens"]),
    ):
        if type(horizon) is not int or horizon <= 0:
            raise ValueError("invalid base horizon")
        rows = []
        for category in CATEGORIES:
            demand = Fraction(horizon * percentages[category], 100)
            if demand.denominator != 1:
                raise ValueError("category demand requires an explicit token rounding rule")
            stock = measured.get(category)
            count = stock["source_specific_stock_tokens"] if stock else None
            rows.append(
                {
                    "category": category,
                    "selected_source": base["initial_sources"][category],
                    "percent": percentages[category],
                    "base_processed_token_demand": int(demand),
                    "source_specific_stock_tokens": count,
                    "optimistic_average_exposure_if_only_this_stock": float(demand / count)
                    if count
                    else None,
                    "joint_training_eligible_tokens": None,
                    "supply_status": "text_and_token_receipts_reconciled_joint_view_pending"
                    if stock
                    else "no_completed_paired_stock_receipts_in_snapshot",
                }
            )
        scenarios.append({"name": name, "base_tokens": horizon, "sources": rows})
    return {
        "format": "speck_base_supply_exposure_report",
        "format_version": 1,
        "manifest": {"path": str(manifest_path), "sha256": file_sha256(manifest_path)},
        "inputs": identities,
        "implementation": {"path": str(Path(__file__).resolve()), "sha256": file_sha256(__file__)},
        "stock_receipts": measured,
        "scenarios": scenarios,
        "training_authority": False,
        "globally_unique_tokens": None,
        "production_ready": False,
        "boundary": "Base-only demand under the selected starting mixture; not an optimized recipe or a repetition recommendation. Ratios assume every token in the single measured source stock remains eligible and no more is acquired, so further view/packing losses raise required exposure. They exclude research, continuation and post-training exposure. Null supply means no completed paired receipt in this snapshot, not zero upstream availability. Stocks and overlapping predecessors are never summed. This check reopens receipt/token-manifest identities, not all corpus shards; full view/materialization verification remains required.",
    }


def render_markdown(result):
    lines = ["# Base-source demand and measured stock", "", result["boundary"], ""]
    for scenario in result["scenarios"]:
        lines += [
            f"## {scenario['name'].capitalize()}: {scenario['base_tokens']:,} base tokens",
            "",
            "| Category | Selected source | Base demand | Measured source stock | Conditional average exposure |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
        for row in scenario["sources"]:
            count = row["source_specific_stock_tokens"]
            exposure = row["optimistic_average_exposure_if_only_this_stock"]
            lines.append(
                f"| {row['category']} | {row['selected_source']} | {row['base_processed_token_demand']:,} | "
                + (f"{count:,}" if count is not None else "pending")
                + " | "
                + (f"{exposure:.2f}×" if exposure is not None else "unmeasured")
                + " |"
            )
        lines.append("")
    return "\n".join(lines)
