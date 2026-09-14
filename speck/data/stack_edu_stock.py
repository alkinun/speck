"""Bind Stack-Edu stock to matched languages, file licenses, and qualified code filters."""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from speck.data.acquisition_units import _bound_identity
from speck.data.sources.stack_v3 import _secret_counts
from speck.data.sources.stack_v3_refine import _english_prose_result
from speck.data.stack_edu_metadata import load_metadata_plan
from speck.experiments.code_languages import load_code_languages
from speck.provenance.io import file_sha256
from speck.tokenization.tokenizer import Tokenizer


def metadata_rejection(row, language, policy):
    filters = policy["filters"]
    if row.get("language") != language:
        raise ValueError("Stack-Edu metadata language differs from its bound unit")
    if not isinstance(row.get("blob_id"), str) or not re.fullmatch(r"[0-9a-f]{40}", row["blob_id"]):
        return "metadata_blob_identity"
    if any(
        not isinstance(row.get(key), str) or not row[key].strip() for key in ("repo_name", "path")
    ):
        return "metadata_attribution"
    if type(row.get("int_score")) is not int or row["int_score"] < filters["minimum_integer_score"]:
        return "metadata_score"
    licenses = row.get("detected_licenses")
    if (
        row.get("license_type") != "permissive"
        or not isinstance(licenses, list)
        or not licenses
        or any(license_id not in filters["accepted_detected_licenses"] for license_id in licenses)
    ):
        return "metadata_license"
    if row.get("src_encoding") not in filters["accepted_encodings"]:
        return "metadata_encoding"
    if (
        type(row.get("length_bytes")) is not int
        or not filters["min_file_bytes"] <= row["length_bytes"] <= filters["max_file_bytes"]
    ):
        return "metadata_size"
    parts = row["path"].replace("\\", "/").lower().split("/")
    if any(part in policy["excluded_path_components"] for part in parts):
        return "declared_vendor_path"
    return None


def decode_code(row, raw, policy):
    if hashlib.sha1(raw).hexdigest() != row["blob_id"]:
        raise ValueError("Stack-Edu blob content identity mismatch")
    if len(raw) != row["length_bytes"]:
        return "content_length_metadata_mismatch", None, None
    try:
        text = raw.decode(row["src_encoding"], errors="strict")
    except (LookupError, UnicodeDecodeError):
        return "content_decode", None, None
    if text.encode("utf-8") != raw:
        return "content_non_identity_utf8", None, None
    if sum(_secret_counts(text).values()):
        return "code_high_confidence_secret", None, None
    prose, probability = _english_prose_result(
        text, row["language"], policy["filters"]["English_prose"]
    )
    if prose == "non_English":
        return "code_non_English_prose", None, None
    return None, text, {"prose_status": prose, "detected_English_probability": probability}


def count_code_tokens(path, tokenizer_identity):
    if file_sha256(tokenizer_identity["path"]) != tokenizer_identity["sha256"]:
        raise ValueError("code counter tokenizer identity mismatch")
    tokenizer = Tokenizer(tokenizer_identity["path"])
    by_language, repositories = {}, Counter()
    documents = tokens = characters = 0
    batch = []

    def consume():
        nonlocal documents, tokens
        encoded = tokenizer.encode_batch([row["text"] for row in batch], bos=True, eos=True)
        for row, ids in zip(batch, encoded, strict=True):
            language = row["language"]
            counts = by_language.setdefault(language, {"documents": 0, "tokens": 0})
            counts["documents"] += 1
            counts["tokens"] += len(ids)
            documents += 1
            tokens += len(ids)
            repositories[row["repo_path"]] += len(row["text"].encode())

    with Path(path).open() as handle:
        for line in handle:
            row = json.loads(line)
            batch.append(row)
            characters += len(row["text"])
            if len(batch) >= 128 or characters >= 1000000:
                consume()
                batch, characters = [], 0
        if batch:
            consume()
    total = sum(repositories.values())
    return {
        "documents": documents,
        "tokens": tokens,
        "tokenizer": tokenizer_identity,
        "by_language": by_language,
        "repositories": len(repositories),
        "repository_byte_hhi": sum((size / total) ** 2 for size in repositories.values())
        if total
        else None,
        "largest_repositories": [
            {"repository": repo, "utf8_bytes": size, "byte_share": size / total}
            for repo, size in sorted(repositories.items(), key=lambda x: (-x[1], x[0]))[:20]
        ],
        "scope": "Frozen-Mistral whole-document tokens including BOS/EOS, grouped by declared language after full exclusion. Repository diagnostics do not reweight the source. No final training view or independence claim.",
    }


def load_stack_edu_preparation(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if (
        value.get("format") != "speck_stack_edu_stock_preparation"
        or value.get("format_version") != 1
        or value.get("source_id") != "stack_edu"
        or value.get("training_authority") is not False
        or value.get("checkpoint_rows") != 256
        or value.get("target_reference_tokens") != 1440000000
        or value.get("candidate_nominal_multiplier") != 2
        or value.get("maximum_observed_wal_bytes") != 2147483648
        or value.get("repository_policy") != "natural_postfilter_no_tokenizer_sample_repository_cap"
        or value.get("excluded_path_components")
        != ["node_modules", "third-party", "third_party", "vendor"]
    ):
        raise ValueError("unsupported Stack-Edu stock plan")
    metadata_id = _bound_identity(value["metadata_result"], path.parent)
    metadata_result = json.loads(Path(metadata_id["path"]).read_text())
    metadata_plan_id = _bound_identity(metadata_result["plan"], Path(metadata_id["path"]).parent)
    metadata_plan = load_metadata_plan(metadata_plan_id["path"])
    if (
        metadata_result.get("format") != "speck_stack_edu_metadata_acquisition_result"
        or metadata_result.get("format_version") != 1
    ):
        raise ValueError("unsupported Stack-Edu metadata result")
    if (
        metadata_result.get("status") != "complete_metadata_verified_not_code_stock"
        or metadata_result.get("training_authority") is not False
    ):
        raise ValueError("Stack-Edu stock requires completed metadata acquisition")
    language_id = _bound_identity(value["code_languages"], path.parent)
    if language_id != metadata_plan["inputs"]["code_languages"]:
        raise ValueError("code languages differ from the completed metadata intake")
    languages = load_code_languages(language_id["path"])
    targets = {
        row["language"]: row
        for row in languages["source_language_capacity_envelope"]
        if row["source_id"] == "stack_edu"
    }
    qualified_id = _bound_identity(value["source_qualification"], path.parent)
    if qualified_id != metadata_plan["inputs"]["source_qualification"]:
        raise ValueError("code qualification differs from the approved metadata intake")
    qualification = json.loads(Path(qualified_id["path"]).read_text())
    base_id = _bound_identity(value["base_plan"], path.parent)
    base_path = Path(base_id["path"])
    base = json.loads(base_path.read_text())
    for key in ("source_registry", "rights_record", "deny_ledger", "contamination_plan"):
        base[key] = _bound_identity(base[key], base_path.parent)
    base["security"]["gitleaks_binary"] = _bound_identity(
        base["security"]["gitleaks_binary"], base_path.parent
    )
    if base["rights_record"]["sha256"] != metadata_plan["inputs"]["source_use"]["sha256"]:
        raise ValueError("Stack-Edu stock and metadata source approvals differ")
    rights = json.loads(Path(base["rights_record"]["path"]).read_text())
    decision_id = _bound_identity(value["tokenizer_decision"], path.parent)
    if decision_id != languages["inputs"]["tokenizer_decision"]:
        raise ValueError("Stack-Edu tokenizer decision differs from its language contract")
    tokenizer_id = _bound_identity(value["reference_tokenizer"], path.parent)
    if tokenizer_id["sha256"] != languages["tokenizer_sha256"]:
        raise ValueError("Stack-Edu counter differs from the frozen tokenizer")
    filters = {
        key: val
        for key, val in qualification["filters"].items()
        if key not in ("language_sample_bytes", "max_bytes_per_language_per_repository")
    }
    base["stack_edu_policy"] = {
        "filters": filters,
        "tokenizer": tokenizer_id,
        "repository_policy": value["repository_policy"],
        "excluded_path_components": value["excluded_path_components"],
        "length_mismatch_policy": "reject",
        "insufficient_prose_policy": "retain_syntax_exempt",
        "fetch": {
            **qualification["fetch"],
            "maximum_blob_bytes": 1000000,
            "maximum_compressed_bytes": 2065536,
        },
    }
    units = []
    expected_units = {unit["id"]: unit for unit in metadata_plan["units"]}
    seen = set()
    for entry in metadata_result["files"]:
        unit = entry["unit"]
        language = unit["language"]
        if unit != expected_units.get(unit["id"]) or language in seen or language not in targets:
            raise ValueError("Stack-Edu metadata unit coverage differs from its plan")
        seen.add(language)
        if (
            entry["raw"]["sha256"] != unit["raw"]["sha256"]
            or entry["verification"]["schema_and_all_row_languages_pass"] is not True
        ):
            raise ValueError("Stack-Edu metadata result is not verified")
        units.append(
            {
                **unit,
                "category": "code",
                "id": f"stack_edu__{unit['reader']['tree_path']}",
                "metadata_path": entry["raw"]["path"],
                "start_row": 0,
                "stop_row": unit["expected_file_rows"],
                "candidate_target_tokens": 2 * targets[language]["nominal_tokens"],
            }
        )
    if seen != set(targets):
        raise ValueError("Stack-Edu stock lacks a matched language")
    return {
        **value,
        "base": base,
        "units": units,
        "source_use": {**rights, "identity": base["rights_record"]},
        "source_language_targets": {
            key: val["preparation_target_tokens"] for key, val in targets.items()
        },
        "raw_directory": metadata_plan["raw_directory"],
        "blob_cache": str((path.parent / value["blob_cache"]).resolve()),
        "output_directory": str((path.parent / value["output_directory"]).resolve()),
    }
