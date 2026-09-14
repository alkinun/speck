"""Prepare the FineMath incumbent with explicit per-document and corpus-selection policies."""

import json
import math
import re
from pathlib import Path

from speck.data.acquisition_units import _bound_identity
from speck.data.configuration import _validate_source
from speck.data.sources.stack_v3 import _secret_counts
from speck.data.sources.web_sample import _host, _raw_pii


def finemath_rejection(document, filters):
    metadata = document["metadata"]
    score = metadata.get("language_score")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
        or not filters["minimum_language_score"] <= score <= 1
    ):
        return "finemath_metadata_language_score"
    text = document["content"]
    if not filters["min_document_bytes"] <= len(text.encode()) <= filters["max_document_bytes"]:
        return "finemath_document_bytes"
    if metadata.get("url") is not None and _host(metadata["url"]) is None:
        return "finemath_invalid_url"
    emails, ips = _raw_pii(
        text,
        {x.lower() for x in filters["allowed_email_placeholders"]},
        set(filters["allowed_ipv4_placeholders"]),
    )
    if emails or ips:
        return "finemath_qualified_PII"
    if sum(_secret_counts(text).values()):
        return "finemath_high_confidence_secret"
    return None


def load_finemath_preparation(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    version = value.get("format_version")
    if (
        value.get("format") != "speck_finemath_stock_preparation"
        or type(version) is not int
        or version not in (1, 2)
        or value.get("source_id") != "finemath_4plus"
        or value.get("training_authority") is not False
        or value.get("checkpoint_rows") != 256
        or value.get("target_reference_tokens") != 960000000
        or value.get("maximum_observed_wal_bytes") != 2147483648
        or value.get("domain_policy")
        != "natural_postfilter_distribution_no_tokenizer_sample_host_cap"
        or value.get("report_domain_concentration") is not True
    ):
        raise ValueError("unsupported FineMath stock plan")
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
        raise ValueError("FineMath requires its existing human source approval")
    registry = json.loads(Path(base["source_registry"]["path"]).read_text())
    source = next(row for row in registry["sources"] if row["id"] == value["source_id"])
    if source["category"] != "math" or any(
        source[key] != qualification["source"][key] for key in ("id", "repo", "revision")
    ):
        raise ValueError("FineMath source differs from its qualified registry view")
    filters = qualification["filters"]
    if (
        filters["required_language"] != "en"
        or filters["minimum_quality_score"] != 4
        or filters["minimum_language_score"] != 0.8
        or filters["language_detector"] != "py3langid==0.3.0"
    ):
        raise ValueError("FineMath source filter contract changed")
    base["finemath_filters"] = {**filters, "maximum_bytes_per_host": None}
    base["math_english"] = {
        "language_policy": "math_prose",
        "minimum_prose_alphabetic_characters": 80,
        "minimum_detected_English_probability": 0.8,
    }
    reader = _validate_source(
        {
            "id": source["id"],
            "repo": source["repo"],
            "revision": source["revision"],
            "tree_path": "finemath-4plus",
            "content_column": "text",
            "file_format": "parquet",
            "language_column": "language",
            "score_column": "int_score",
            "filters": {"language": "en", "min_score": 4},
            "metadata_columns": {
                "url": "url",
                "language_score": "language_score",
                "math_markup": "metadata",
                "crawl": "crawl",
            },
        }
    )
    shard_id = _bound_identity(value["shard_manifest"], path.parent)
    shards = json.loads(Path(shard_id["path"]).read_text())
    file_count = 8 if version == 1 else 11
    if (
        shards.get("format") != "speck_finemath_shard_manifest"
        or shards.get("format_version") != 1
        or any(shards[key] != source[key] for key in ("repo", "revision"))
        or [row["filename"] for row in shards["files"]]
        != [f"finemath-4plus/train-{i:05d}-of-00064.parquet" for i in range(file_count)]
    ):
        raise ValueError("FineMath stock must use its version's pinned complete shards")
    if version == 2:
        previous_id = _bound_identity(value["predecessor_plan"], path.parent)
        previous_path = Path(previous_id["path"])
        previous = json.loads(previous_path.read_text())
        if previous.get("format_version") != 1:
            raise ValueError("FineMath v2 requires the eight-shard v1 predecessor")
        changed_fields = {"format_version", "shard_manifest", "output_directory", "scope"}
        for key, original in previous.items():
            if key in changed_fields:
                continue
            current = value.get(key)
            if isinstance(original, dict) and set(original) == {"path", "sha256"}:
                original = _bound_identity(original, previous_path.parent)
                current = _bound_identity(current, path.parent)
            if current != original:
                raise ValueError(f"FineMath successor changes predecessor policy: {key}")
        previous_shard_id = _bound_identity(previous["shard_manifest"], previous_path.parent)
        previous_shards = json.loads(Path(previous_shard_id["path"]).read_text())
        if shards["files"][:8] != previous_shards["files"]:
            raise ValueError("FineMath successor changes the original shard prefix")
        old_output = (previous_path.parent / previous["output_directory"]).resolve()
        new_output = (path.parent / value["output_directory"]).resolve()
        if old_output.is_relative_to(new_output) or new_output.is_relative_to(old_output):
            raise ValueError("FineMath successor must use a separate output directory")
        result_id = _bound_identity(value["predecessor_result"], path.parent)
        result = json.loads(Path(result_id["path"]).read_text())
        if (
            result.get("format") != "speck_finemath_stock_preparation_result"
            or result["plan"]["sha256"] != previous_id["sha256"]
            or result.get("training_authority") is not False
            or result.get("capacity_target_pass") is not False
            or result.get("storage_gate_pass") is not True
        ):
            raise ValueError("FineMath successor requires the completed shortfall result")
        expected_units = {f"finemath_4plus__file_{i}" for i in range(8)}
        if set(value.get("reuse_acquisition_units", {})) != expected_units:
            raise ValueError("FineMath successor must bind all eight completed acquisition units")
    units = []
    for i, row in enumerate(shards["files"]):
        if (
            type(row["rows"]) is not int
            or not 1 <= row["rows"] <= 200000
            or type(row["bytes"]) is not int
            or not 0 < row["bytes"] <= 1_000_000_000
            or not isinstance(row["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
        ):
            raise ValueError("FineMath shard rows exceed the declared envelope")
        units.append(
            {
                "id": f"finemath_4plus__file_{i}",
                "category": "math",
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
        raise ValueError("FineMath counter differs from the frozen base tokenizer")
    return {
        **value,
        "base": base,
        "units": units,
        "source_use": {**rights, "identity": base["rights_record"]},
        "raw_directory": str((path.parent / value["raw_directory"]).resolve()),
        "output_directory": str((path.parent / value["output_directory"]).resolve()),
    }
