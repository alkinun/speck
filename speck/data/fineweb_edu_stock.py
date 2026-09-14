"""Prepare the FineWeb-Edu incumbent with frozen web filters and explicit host policy."""

import json
import math
import re
from pathlib import Path

from speck.data.acquisition_units import _bound_identity
from speck.data.configuration import _validate_source
from speck.data.sources.stack_v3 import _secret_counts
from speck.data.sources.stack_v3_refine import _language_identifier
from speck.data.sources.web_sample import _duplicate_line_ratio, _host, _raw_pii


def fineweb_edu_rejection(document, filters):
    metadata, text = document["metadata"], document["content"]
    if not isinstance(metadata.get("id"), str) or not metadata["id"].strip():
        return "web_missing_document_id", None
    if metadata.get("language") != filters["required_language"]:
        return "web_metadata_language", None
    confidence = metadata.get("language_score")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not filters["minimum_language_score"] <= confidence <= 1
    ):
        return "web_metadata_language_score", None
    score = metadata.get("quality_score")
    if type(score) is not int or not filters["minimum_quality_score"] <= score <= 5:
        return "web_metadata_quality_score", None
    if not filters["min_document_bytes"] <= len(text.encode()) <= filters["max_document_bytes"]:
        return "web_document_bytes", None
    if (
        sum(char.isalpha() for char in text) / max(len(text), 1)
        < filters["minimum_alphabetic_ratio"]
    ):
        return "web_alphabetic_ratio", None
    if _duplicate_line_ratio(text) > filters["maximum_duplicate_line_ratio"]:
        return "web_duplicate_lines", None
    host = _host(metadata.get("url"))
    if host is None:
        return "web_invalid_url", None
    if any(term in host for term in filters["adult_host_terms"]):
        return "web_excluded_host", None
    emails, ips = _raw_pii(
        text,
        {x.lower() for x in filters["allowed_email_placeholders"]},
        set(filters["allowed_ipv4_placeholders"]),
    )
    if emails or ips:
        return "web_qualified_PII", None
    if sum(_secret_counts(text).values()):
        return "web_high_confidence_secret", None
    language, probability = _language_identifier().classify(text)
    if language != "en" or probability < filters["minimum_detected_English_probability"]:
        return "web_detected_non_English", float(probability)
    return None, float(probability)


def load_fineweb_edu_preparation(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    version = value.get("format_version")
    if (
        value.get("format") != "speck_fineweb_edu_stock_preparation"
        or type(version) is not int
        or version != 1
        or value.get("source_id") != "fineweb_edu"
        or value.get("training_authority") is not False
        or value.get("checkpoint_rows") != 256
        or value.get("target_reference_tokens") != 5280000000
        or value.get("maximum_observed_wal_bytes") != 2147483648
        or value.get("domain_policy")
        != "natural_postfilter_distribution_no_tokenizer_sample_host_cap"
        or value.get("report_domain_concentration") is not True
    ):
        raise ValueError("unsupported FineWeb-Edu stock plan")
    qualified = _bound_identity(value["source_qualification"], path.parent)
    qualification = json.loads(Path(qualified["path"]).read_text())
    base_id = _bound_identity(value["base_plan"], path.parent)
    base_path = Path(base_id["path"])
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
        raise ValueError("FineWeb-Edu requires its existing human source approval")
    registry = json.loads(Path(base["source_registry"]["path"]).read_text())
    source = next(row for row in registry["sources"] if row["id"] == value["source_id"])
    if source["category"] != "web" or any(
        source[key] != qualification["source"][key] for key in ("id", "repo", "revision")
    ):
        raise ValueError("FineWeb-Edu source differs from its qualified registry view")
    filters = qualification["filters"]
    if (
        filters["required_language"] != "en"
        or filters["minimum_quality_score"] != 3
        or filters["minimum_language_score"] != 0.8
        or filters["language_detector"] != "py3langid==0.3.0"
    ):
        raise ValueError("FineWeb-Edu source filter contract changed")
    base["fineweb_edu_filters"] = {**filters, "maximum_bytes_per_host": None}
    reader = _validate_source(
        {
            "id": source["id"],
            "repo": source["repo"],
            "revision": source["revision"],
            "tree_path": "sample/10BT",
            "content_column": "text",
            "file_format": "parquet",
            "language_column": "language",
            "score_column": "int_score",
            "filters": {"language": "en", "min_score": 3},
            "metadata_columns": {
                "id": "id",
                "url": "url",
                "language": "language",
                "language_score": "language_score",
                "quality_score": "int_score",
                "crawl": "dump",
                "file_path": "file_path",
            },
        }
    )
    shard_id = _bound_identity(value["shard_manifest"], path.parent)
    shards = json.loads(Path(shard_id["path"]).read_text())
    required_columns = {
        "text",
        "id",
        "url",
        "language",
        "language_score",
        "int_score",
        "dump",
        "file_path",
    }
    if (
        shards.get("format") != "speck_fineweb_edu_shard_manifest"
        or shards.get("format_version") != 1
        or any(shards[key] != source[key] for key in ("repo", "revision"))
        or [row["filename"] for row in shards["files"]]
        != [f"sample/10BT/{i:03d}_00000.parquet" for i in range(14)]
        or any(not required_columns.issubset(row["columns"]) for row in shards["files"])
    ):
        raise ValueError("FineWeb-Edu stock must use all fourteen pinned complete 10BT files")
    if shards["files"][-1]["sha256"] != qualification["source"]["sha256"]:
        raise ValueError("FineWeb-Edu stock does not include the original qualified shard")
    units = []
    for i, row in enumerate(shards["files"]):
        if (
            type(row["rows"]) is not int
            or not 1 <= row["rows"] <= 1000000
            or type(row["bytes"]) is not int
            or not 0 < row["bytes"] <= 3_000_000_000
            or not isinstance(row["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
        ):
            raise ValueError("FineWeb-Edu shard rows exceed the declared envelope")
        units.append(
            {
                "id": f"fineweb_edu__file_{i}",
                "category": "web",
                "reader": reader,
                "raw": {
                    "source_id": source["id"],
                    **{key: row[key] for key in ("filename", "bytes", "sha256")},
                },
                "start_row": 0,
                "stop_row": row["rows"],
                "expected_file_rows": row["rows"],
            }
        )
    decision_id = _bound_identity(value["tokenizer_decision"], path.parent)
    decision = json.loads(Path(decision_id["path"]).read_text())
    if (
        decision["status"] != "tokenizer_selected_and_frozen"
        or decision["tokenizer_fingerprint"] != value["reference_tokenizer"]["sha256"]
    ):
        raise ValueError("FineWeb-Edu counter differs from the frozen base tokenizer")
    return {
        **value,
        "base": base,
        "units": units,
        "source_use": {**rights, "identity": base["rights_record"]},
        "raw_directory": str((path.parent / value["raw_directory"]).resolve()),
        "output_directory": str((path.parent / value["output_directory"]).resolve()),
    }
