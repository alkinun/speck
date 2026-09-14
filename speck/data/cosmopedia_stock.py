"""Prepare Cosmopedia generated text with prompt lineage and qualified document filters."""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from speck.data.acquisition_units import _bound_identity
from speck.data.configuration import _validate_source
from speck.data.sources.stack_v3 import _secret_counts
from speck.data.sources.stack_v3_refine import _language_identifier
from speck.data.sources.synthetic_sample import (
    _repeated_ngram_ratio,
    _shingle_jaccard,
    _template_prefix,
)
from speck.data.sources.web_sample import _duplicate_line_ratio, _raw_pii


def cosmopedia_document(document, policy):
    """Return a rejection reason and lineage without retaining raw prompts as training text."""
    text, metadata = document["content"], document["metadata"]
    filters = policy["filters"]
    if not filters["min_document_bytes"] <= len(text.encode()) <= filters["max_document_bytes"]:
        return "synthetic_document_bytes", None
    if _duplicate_line_ratio(text) > filters["maximum_duplicate_line_ratio"]:
        return "synthetic_duplicate_lines", None
    repetition = _repeated_ngram_ratio(text, filters["repetition_ngram_tokens"])
    if repetition > filters["maximum_repeated_ngram_ratio"]:
        return "synthetic_repeated_ngrams", None
    if any(phrase.lower() in text.lower() for phrase in filters["model_identity_phrases"]):
        return "synthetic_model_identity_phrase", None
    emails, ips = _raw_pii(
        text,
        {value.lower() for value in filters["allowed_email_placeholders"]},
        set(filters["allowed_ipv4_placeholders"]),
    )
    if emails or ips:
        return "synthetic_qualified_PII", None
    if sum(_secret_counts(text).values()):
        return "synthetic_high_confidence_secret", None
    if any(
        not isinstance(metadata.get(key), str) or not metadata[key].strip()
        for key in ("prompt", "style", "seed_source_label")
    ):
        return "synthetic_missing_lineage", None
    detected, probability = _language_identifier().classify(text)
    if detected != "en" or probability < filters["minimum_detected_English_probability"]:
        return "synthetic_non_English", None
    template = _template_prefix(text, filters["template_prefix_tokens"])
    if not template:
        return "synthetic_empty_template", None
    overlap = _shingle_jaccard(text, metadata["prompt"], filters["seed_overlap_shingle_tokens"])
    if overlap >= policy["maximum_seed_text_jaccard"]:
        return "synthetic_unchanged_seed", None
    prompt_hash = hashlib.sha256(metadata["prompt"].encode()).hexdigest()
    return None, {
        "generator": policy["generator"],
        "seed_source": policy["seed_source"],
        "transformation": policy["transformation"],
        "seed_source_label": metadata["seed_source_label"],
        "style": metadata["style"],
        "audience": metadata.get("audience"),
        "prompt_sha256": prompt_hash,
        "seed_sha256": prompt_hash,
        "seed_hash_scope": "full_released_prompt_including_web_excerpt_not_original_seed_identity",
        "seed_text_shingle_jaccard": overlap,
        "detected_English_probability": float(probability),
        "repeated_ngram_ratio": repetition,
        "template_prefix_sha256": hashlib.sha256(template.encode()).hexdigest(),
    }


def cosmopedia_diagnostics(path):
    styles, labels, templates, prompts = (Counter() for _ in range(4))
    documents = total_bytes = 0
    maximum_overlap = 0.0
    with Path(path).open() as handle:
        for raw in handle:
            record = json.loads(raw)
            metadata = record["metadata"]
            size = len(record["text"].encode())
            documents += 1
            total_bytes += size
            styles[metadata["style"]] += size
            labels[metadata["seed_source_label"]] += size
            templates[metadata["template_prefix_sha256"]] += size
            prompts[metadata["prompt_sha256"]] += 1
            maximum_overlap = max(maximum_overlap, metadata["seed_text_shingle_jaccard"])

    def shares(counter):
        return [
            {"label": key, "utf8_bytes": value, "byte_share": value / total_bytes}
            for key, value in sorted(counter.items(), key=lambda x: (-x[1], x[0]))[:20]
        ]

    return {
        "documents": documents,
        "utf8_bytes": total_bytes,
        "styles": shares(styles),
        "seed_source_labels": shares(labels),
        "template_prefixes": len(templates),
        "largest_template_prefixes": shares(templates),
        "template_byte_hhi": sum((value / total_bytes) ** 2 for value in templates.values())
        if total_bytes
        else None,
        "distinct_prompt_hashes": len(prompts),
        "documents_beyond_first_identical_prompt": sum(count - 1 for count in prompts.values()),
        "maximum_generated_text_prompt_jaccard": maximum_overlap if documents else None,
        "seed_domain_coverage": "unavailable_in_released_schema",
        "boundary": "Natural postfilter distribution diagnostics, without sampler byte quotas or reweighting. Prompt hashes are not original seed IDs or proof of independent ancestry; seed_data is a source label. Generator and seed revisions remain undisclosed. No correctness or model-quality claim.",
    }


def load_cosmopedia_preparation(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if (
        value.get("format") != "speck_cosmopedia_stock_preparation"
        or value.get("format_version") != 1
        or value.get("source_id") != "cosmopedia_v2"
        or value.get("training_authority") is not False
        or value.get("checkpoint_rows") != 256
        or value.get("target_reference_tokens") != 960000000
        or value.get("maximum_observed_wal_bytes") != 2147483648
        or value.get("corpus_policy") != "natural_postfilter_no_sampler_template_or_domain_quota"
        or value.get("report_synthetic_concentration") is not True
    ):
        raise ValueError("unsupported Cosmopedia stock plan")
    qualified_id = _bound_identity(value["source_qualification"], path.parent)
    qualification = json.loads(Path(qualified_id["path"]).read_text())
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
        raise ValueError("Cosmopedia requires its existing human source approval")
    registry = json.loads(Path(base["source_registry"]["path"]).read_text())
    source = next(row for row in registry["sources"] if row["id"] == value["source_id"])
    if source["category"] != "synthetic" or any(
        source[key] != qualification["source"][key] for key in ("id", "repo", "revision")
    ):
        raise ValueError("Cosmopedia differs from its qualified registry view")
    item = qualification["source"]["inputs"][0]
    if (
        item["fields"]["seed"] != "prompt"
        or item["fields"]["seed_label"] != "seed_data"
        or item["maximum_seed_text_jaccard"] != 0.8
        or qualification["filters"]["language_detector"] != "py3langid==0.3.0"
    ):
        raise ValueError("Cosmopedia requires corrected prompt/seed-label qualification")
    base["cosmopedia_policy"] = {
        "filters": {
            **qualification["filters"],
            "maximum_bytes_per_template_prefix": None,
            "maximum_bytes_per_seed_domain": None,
        },
        **{
            key: item[key]
            for key in ("generator", "seed_source", "transformation", "maximum_seed_text_jaccard")
        },
        "corpus_policy": value["corpus_policy"],
        "require_prompt_style_seed_label": True,
    }
    reader = _validate_source(
        {
            "id": source["id"],
            "repo": source["repo"],
            "revision": source["revision"],
            "tree_path": "cosmopedia-v2",
            "file_format": "parquet",
            "content_column": "text",
            "metadata_columns": {
                "prompt": "prompt",
                "style": "format",
                "seed_source_label": "seed_data",
                "audience": "audience",
            },
            "filters": {},
        }
    )
    shard_id = _bound_identity(value["shard_manifest"], path.parent)
    shards = json.loads(Path(shard_id["path"]).read_text())
    if (
        shards.get("format") != "speck_cosmopedia_shard_manifest"
        or shards.get("format_version") != 1
        or any(shards[key] != source[key] for key in ("repo", "revision"))
        or [row["filename"] for row in shards["files"]]
        != [f"cosmopedia-v2/train-{i:05d}-of-00104.parquet" for i in range(5)]
    ):
        raise ValueError("Cosmopedia stock requires five complete pinned shards")
    units = []
    for i, row in enumerate(shards["files"]):
        if (
            type(row["rows"]) is not int
            or not 1 <= row["rows"] <= 500000
            or type(row["bytes"]) is not int
            or not 0 < row["bytes"] <= 2000000000
            or not isinstance(row["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
            or not (set(reader["metadata_columns"].values()) | {"text"}).issubset(row["columns"])
        ):
            raise ValueError("Cosmopedia shard exceeds its row/size/schema envelope")
        units.append(
            {
                "id": f"cosmopedia_v2__file_{i}",
                "category": "synthetic",
                "reader": reader,
                "raw": {
                    "source_id": source["id"],
                    **{k: row[k] for k in ("filename", "bytes", "sha256")},
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
        raise ValueError("Cosmopedia counter differs from the frozen tokenizer")
    return {
        **value,
        "base": base,
        "units": units,
        "source_use": {**rights, "identity": base["rights_record"]},
        "raw_directory": str((path.parent / value["raw_directory"]).resolve()),
        "output_directory": str((path.parent / value["output_directory"]).resolve()),
    }
