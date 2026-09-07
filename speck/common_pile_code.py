"""Build a bounded code sample from a pinned Common Pile filtered shard."""

import gzip
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from speck.stack_v3 import _secret_counts
from speck.stack_v3_refine import _english_prose_result

FORMAT = "speck_common_pile_code_sample"
FORMAT_VERSION = 1
REPORT_FORMAT = "speck_common_pile_code_sample_result"


def _sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _exact_keys(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{name} must contain exactly: {', '.join(sorted(keys))}")


def _path(value, name, config_dir):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(value).expanduser()
    return str((config_dir / path).resolve() if not path.is_absolute() else path.resolve())


def _digest(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def validate_common_pile_code_config(config, *, config_dir=None):
    """Validate a bounded Common Pile code sample plan."""

    config_dir = Path(config_dir or ".").resolve()
    _exact_keys(
        config,
        {"format", "format_version", "status", "source", "filters", "output_directory"},
        "Common Pile code sample",
    )
    if config["format"] != FORMAT or config["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported Common Pile code sample format")
    if config["status"] != "bounded_sample_authorized_not_training_authority":
        raise ValueError("Common Pile code sample must remain non-authoritative")
    source = config["source"]
    _exact_keys(
        source,
        {"repo", "revision", "official_url", "path", "sha256", "rows", "source_value"},
        "source",
    )
    normalized_source = {
        **source,
        "path": _path(source["path"], "source.path", config_dir),
        "sha256": _digest(source["sha256"], "source.sha256"),
        "rows": _integer(source["rows"], "source.rows", 1),
    }
    for key in ("repo", "revision", "official_url", "source_value"):
        if not isinstance(source[key], str) or not source[key]:
            raise ValueError(f"source.{key} must be non-empty")

    filters = config["filters"]
    _exact_keys(
        filters,
        {
            "language",
            "minimum_integer_score",
            "license_type",
            "accepted_detected_licenses",
            "encoding",
            "exclude_vendor",
            "exclude_generated",
            "min_file_bytes",
            "max_file_bytes",
            "max_bytes_per_repository",
            "sample_bytes",
            "English_prose",
        },
        "filters",
    )
    if filters["license_type"] != "permissive":
        raise ValueError("Common Pile code must use permissive rows")
    for key in ("language", "encoding"):
        if not isinstance(filters[key], str) or not filters[key]:
            raise ValueError(f"filters.{key} must be non-empty")
    for key in ("exclude_vendor", "exclude_generated"):
        if not isinstance(filters[key], bool):
            raise ValueError(f"filters.{key} must be boolean")
    licenses = filters["accepted_detected_licenses"]
    if (
        not isinstance(licenses, list)
        or not licenses
        or any(not isinstance(value, str) or not value for value in licenses)
        or len(licenses) != len(set(licenses))
    ):
        raise ValueError("accepted_detected_licenses must be unique strings")
    minimum = _integer(filters["min_file_bytes"], "min_file_bytes", 1)
    maximum = _integer(filters["max_file_bytes"], "max_file_bytes", 1)
    if minimum > maximum:
        raise ValueError("minimum file bytes cannot exceed maximum")
    english = filters["English_prose"]
    _exact_keys(
        english,
        {"detector", "minimum_alphabetic_characters", "minimum_probability"},
        "English_prose",
    )
    if english["detector"] != "py3langid==0.3.0":
        raise ValueError("English prose detector must be pinned py3langid")
    probability = english["minimum_probability"]
    if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not 0 < probability <= 1:
        raise ValueError("English probability must be in (0, 1]")
    normalized = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "status": config["status"],
        "source": normalized_source,
        "filters": {
            **filters,
            "minimum_integer_score": _integer(
                filters["minimum_integer_score"], "minimum_integer_score", 1
            ),
            "accepted_detected_licenses": sorted(licenses),
            "min_file_bytes": minimum,
            "max_file_bytes": maximum,
            "max_bytes_per_repository": _integer(
                filters["max_bytes_per_repository"], "max_bytes_per_repository", 1
            ),
            "sample_bytes": _integer(filters["sample_bytes"], "sample_bytes", 1),
            "English_prose": {
                **english,
                "minimum_alphabetic_characters": _integer(
                    english["minimum_alphabetic_characters"],
                    "minimum_alphabetic_characters",
                    1,
                ),
                "minimum_probability": float(probability),
            },
        },
        "output_directory": _path(config["output_directory"], "output_directory", config_dir),
    }
    normalized["plan_fingerprint"] = _fingerprint(normalized)
    return normalized


def load_common_pile_code_config(path):
    path = Path(path).resolve()
    return validate_common_pile_code_config(
        json.loads(path.read_text(encoding="utf-8")), config_dir=path.parent
    )


def _validated_config(config):
    if "plan_fingerprint" not in config:
        return validate_common_pile_code_config(config)
    payload = {key: value for key, value in config.items() if key != "plan_fingerprint"}
    if config["plan_fingerprint"] != _fingerprint(payload):
        raise ValueError("normalized Common Pile code plan fingerprint mismatch")
    return config


def _dump_line(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def sample_common_pile_code(config, *, restart=False):
    """Extract a bounded, filtered inline Common Pile code sample."""

    config = _validated_config(config)
    source_path = Path(config["source"]["path"])
    if not source_path.is_file() or _sha256(source_path) != config["source"]["sha256"]:
        raise ValueError("Common Pile source identity mismatch")
    output = Path(config["output_directory"])
    staging = output.with_name(output.name + ".building")
    if output.exists():
        raise FileExistsError(f"Common Pile code sample already exists: {output}")
    if staging.exists():
        if not restart:
            raise FileExistsError(f"incomplete Common Pile code sample exists: {staging}")
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    tokenizer_path = staging / "tokenizer-input.jsonl"
    attribution_path = staging / "attribution.jsonl"
    counts = Counter()
    licenses = Counter()
    per_repo = Counter()
    seen = set()
    sampled_bytes = 0
    rows = 0
    with (
        gzip.open(source_path, "rt", encoding="utf-8") as source,
        tokenizer_path.open("w", encoding="utf-8") as tokenizer_output,
        attribution_path.open("w", encoding="utf-8") as attribution_output,
    ):
        for line in source:
            rows += 1
            value = json.loads(line)
            if sampled_bytes >= config["filters"]["sample_bytes"]:
                continue
            counts["rows_seen_before_quota"] += 1
            metadata = value.get("metadata") or {}
            text = value.get("text")
            if value.get("source") != config["source"]["source_value"] or not isinstance(text, str):
                counts["source_or_content_rejected"] += 1
                continue
            if value.get("int_score", 0) < config["filters"]["minimum_integer_score"]:
                counts["score_rejected"] += 1
                continue
            if metadata.get("language") != config["filters"]["language"]:
                counts["language_rejected"] += 1
                continue
            if metadata.get("license_type") != config["filters"]["license_type"]:
                counts["license_type_rejected"] += 1
                continue
            detected = metadata.get("detected_licenses") or []
            if not detected or any(
                license_id not in config["filters"]["accepted_detected_licenses"]
                for license_id in detected
            ):
                counts["license_allowlist_rejected"] += 1
                continue
            if config["filters"]["exclude_vendor"] and metadata.get("is_vendor"):
                counts["vendor_rejected"] += 1
                continue
            if config["filters"]["exclude_generated"] and metadata.get("is_generated"):
                counts["generated_rejected"] += 1
                continue
            if metadata.get("src_encoding") != config["filters"]["encoding"]:
                counts["encoding_rejected"] += 1
                continue
            raw = text.encode()
            size = len(raw)
            if not config["filters"]["min_file_bytes"] <= size <= config["filters"][
                "max_file_bytes"
            ]:
                counts["size_rejected"] += 1
                continue
            blob_id = metadata.get("blob_id")
            if (
                not isinstance(blob_id, str)
                or len(blob_id) != 40
                or any(character not in "0123456789abcdef" for character in blob_id)
            ):
                counts["upstream_blob_identity_rejected"] += 1
                continue
            if hashlib.sha1(raw).hexdigest() != blob_id:
                counts["upstream_blob_id_not_plain_released_sha1"] += 1
            if metadata.get("length_bytes") != size:
                counts["content_length_metadata_mismatch"] += 1
            released_sha256 = hashlib.sha256(raw).hexdigest()
            if released_sha256 in seen:
                counts["exact_duplicate_rejected"] += 1
                continue
            secret_hits = _secret_counts(text)
            if sum(secret_hits.values()):
                counts["high_confidence_secret_rejected"] += 1
                continue
            prose_result, _ = _english_prose_result(
                text, config["filters"]["language"], config["filters"]["English_prose"]
            )
            counts[f"prose_{prose_result}"] += 1
            if prose_result == "non_English":
                counts["non_English_prose_rejected"] += 1
                continue
            repo = metadata.get("repo_name")
            if not isinstance(repo, str) or not repo:
                counts["repository_identity_rejected"] += 1
                continue
            if per_repo[repo] + size > config["filters"]["max_bytes_per_repository"]:
                counts["repository_cap_rejected"] += 1
                continue
            seen.add(released_sha256)
            record = {
                "text": text,
                "source": "common_pile_stackv2_edu",
                "content_id": blob_id,
                "released_content_sha256": released_sha256,
                "repo_path": repo,
                "repo_id": metadata.get("github_id"),
                "commit_id": metadata.get("revision_id"),
                "file_path": metadata.get("path"),
                "language": metadata.get("language"),
                "detected_licenses": detected,
                "size_bytes": size,
                "declared_length_bytes": metadata.get("length_bytes"),
                "score": value.get("score"),
                "int_score": value.get("int_score"),
            }
            _dump_line(tokenizer_output, record)
            _dump_line(
                attribution_output,
                {key: item for key, item in record.items() if key != "text"},
            )
            sampled_bytes += size
            per_repo[repo] += size
            licenses.update(detected)
            counts["records_sampled"] += 1
        for handle in (tokenizer_output, attribution_output):
            handle.flush()
            os.fsync(handle.fileno())
    if rows != config["source"]["rows"]:
        raise ValueError(f"Common Pile row count {rows} != {config['source']['rows']}")
    missing = sampled_bytes < config["filters"]["sample_bytes"]
    report = {
        "format": REPORT_FORMAT,
        "format_version": FORMAT_VERSION,
        "status": (
            "bounded_sample_incomplete_not_training_authority"
            if missing
            else "bounded_sample_complete_not_training_authority"
        ),
        "plan_fingerprint": config["plan_fingerprint"],
        "source": config["source"],
        "filters": config["filters"],
        "counts": {**dict(sorted(counts.items())), "source_rows": rows},
        "repositories": len(per_repo),
        "sampled_bytes": sampled_bytes,
        "target_bytes": config["filters"]["sample_bytes"],
        "accepted_licenses": dict(licenses.most_common()),
        "outputs": {
            "tokenizer_input": {
                "path": tokenizer_path.name,
                "bytes": tokenizer_path.stat().st_size,
                "sha256": _sha256(tokenizer_path),
            },
            "attribution": {
                "path": attribution_path.name,
                "bytes": attribution_path.stat().st_size,
                "sha256": _sha256(attribution_path),
            },
        },
        "gates": {
            "source_identity_and_rows": "pass",
            "released_content_sha256_identity": "pass_for_sampled_records",
            "upstream_blob_identity": "preserved_and_plain_sha1_mismatches_reported",
            "score_license_vendor_generated_filter": "pass",
            "English_comments": "pass_for_declared_extractor_with_strings_out_of_scope",
            "bounded_yield": "fail" if missing else "pass",
            "dedicated_secret_scanner": "pending",
            "benchmark_contamination": "pending",
            "cross_source_near_duplicates": "pending",
            "manual_legal_acceptance": "pending",
            "training_authority": "blocked",
        },
    }
    _write_json(staging / "report.json", report)
    if missing:
        raise RuntimeError("Common Pile source exhausted before sample byte target")
    os.replace(staging, output)
    return report
