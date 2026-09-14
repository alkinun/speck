"""Prepare the approved peS2o background with its qualified language/license/text filters."""

import json
from pathlib import Path

from speck.data.acquisition_units import _bound_identity
from speck.data.configuration import _validate_source
from speck.data.sources.science_sample import _SPACED_OCR, _boilerplate_ratio, _science_hits
from speck.data.sources.stack_v3 import _secret_counts
from speck.data.sources.stack_v3_refine import _language_identifier
from speck.data.sources.web_sample import _duplicate_line_ratio, _raw_pii


def science_rejection(document, filters):
    """Apply the stateless peS2o qualification criteria; dedup happens downstream."""
    text = document["content"]
    if document["metadata"].get("license") not in filters["accepted_document_licenses"]:
        return "science_document_license", None
    if not filters["min_document_bytes"] <= len(text.encode()) <= filters["max_document_bytes"]:
        return "science_document_bytes", None
    if sum(c.isalpha() for c in text) / max(len(text), 1) < filters["minimum_alphabetic_ratio"]:
        return "science_alphabetic_ratio", None
    if _duplicate_line_ratio(text) > filters["maximum_duplicate_line_ratio"]:
        return "science_duplicate_lines", None
    if len(_SPACED_OCR.findall(text)) > filters["maximum_spaced_OCR_sequences"]:
        return "science_spaced_OCR", None
    if text.count("�") / max(len(text), 1) > filters["maximum_replacement_character_ratio"]:
        return "science_replacement_characters", None
    if (
        _boilerplate_ratio(text, filters["boilerplate_line_terms"])
        > filters["maximum_boilerplate_line_ratio"]
    ):
        return "science_boilerplate", None
    if _science_hits(text, filters["science_terms"]) < filters["minimum_science_term_hits"]:
        return "science_term_hits", None
    emails, ips = _raw_pii(
        text,
        {x.lower() for x in filters["allowed_email_placeholders"]},
        set(filters["allowed_ipv4_placeholders"]),
    )
    if emails or ips:
        return "science_raw_PII", None
    if sum(_secret_counts(text).values()):
        return "science_high_confidence_secret", None
    detected, probability = _language_identifier().classify(text)
    if detected != "en" or probability < filters["minimum_detected_English_probability"]:
        return "science_non_English", float(probability)
    return None, float(probability)


def load_science_preparation(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if (
        value.get("format") != "speck_science_stock_preparation"
        or value.get("format_version") != 1
        or value.get("source_id") != "pes2o_v3"
        or value.get("training_authority") is not False
        or value.get("checkpoint_rows") != 256
        or value.get("target_reference_tokens") != 400_000_000
        or value.get("maximum_observed_wal_bytes") != 2_147_483_648
    ):
        raise ValueError("unsupported science stock plan")
    qualified = _bound_identity(value["source_qualification"], path.parent)
    source_plan = json.loads(Path(qualified["path"]).read_text())
    identity = _bound_identity(value["base_plan"], path.parent)
    base_path = Path(identity["path"])
    base = json.loads(base_path.read_text())
    for key in ("source_registry", "rights_record", "deny_ledger", "contamination_plan"):
        base[key] = _bound_identity(base[key], base_path.parent)
    base["security"]["gitleaks_binary"] = _bound_identity(
        base["security"]["gitleaks_binary"], base_path.parent
    )
    rights = json.loads(Path(base["rights_record"]["path"]).read_text())
    if (
        rights.get("status") != "all_sources_human_approved"
        or rights.get("automated_approval_made") is not False
        or value["source_id"] not in rights["approved_source_ids"]
    ):
        raise ValueError("science source needs its existing human approval")
    source = source_plan["source"]
    registry = json.loads(Path(base["source_registry"]["path"]).read_text())
    registered = next(item for item in registry["sources"] if item["id"] == value["source_id"])
    if (
        source["id"] != value["source_id"]
        or registered["category"] != "science"
        or any(source[key] != registered[key] for key in ("repo", "revision"))
        or source["fields"]["text"] != "text"
        or source["file_format"] != "jsonl_zstd"
    ):
        raise ValueError("science source differs from its approved qualified view")
    base["science_filters"] = source_plan["filters"]
    if (
        any(
            source_plan["filters"][key] is not None
            for key in (
                "required_language",
                "minimum_language_score",
                "minimum_quality_score",
                "maximum_bytes_per_host",
            )
        )
        or source_plan["filters"]["language_detector"] != "py3langid==0.3.0"
    ):
        raise ValueError("science stock supports the qualified peS2o filter contract only")
    reader = _validate_source(
        {
            "id": source["id"],
            "repo": source["repo"],
            "revision": source["revision"],
            "tree_path": "data/v3",
            "file_format": "jsonl_zstd",
            "content_column": "text",
            "metadata_columns": {
                "id": "id",
                "url": "metadata.oa_url",
                "license": "metadata.oa_license",
                "created": "created",
                "source_partition": "source",
            },
            "filters": {},
        }
    )
    expected = {
        "filename": "data/v3/train-0040-of-0136.zst",
        "bytes": 983683365,
        "sha256": source["sha256"],
        "rows": source["rows"],
    }
    if value["raw_files"] != [expected] or source["rows"] != 110183:
        raise ValueError("science preparation requires its complete qualified shard")
    unit = {
        "id": "pes2o_v3__file_40",
        "category": "science",
        "reader": reader,
        "raw": {key: item for key, item in expected.items() if key != "rows"}
        | {"source_id": source["id"]},
        "start_row": 0,
        "stop_row": source["rows"],
        "expected_file_rows": source["rows"],
    }
    decision_id = _bound_identity(value["tokenizer_decision"], path.parent)
    decision = json.loads(Path(decision_id["path"]).read_text())
    if (
        decision["status"] != "tokenizer_selected_and_frozen"
        or decision["tokenizer_fingerprint"] != value["reference_tokenizer"]["sha256"]
    ):
        raise ValueError("science counter differs from the selected tokenizer")
    return {
        **value,
        "base": base,
        "units": [unit],
        "source_use": {**rights, "identity": base["rights_record"]},
        "raw_directory": str((path.parent / value["raw_directory"]).resolve()),
        "output_directory": str((path.parent / value["output_directory"]).resolve()),
    }
