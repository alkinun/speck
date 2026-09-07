"""Fetch a bounded, license-filtered MegaMath code sample from Software Heritage."""

import hashlib
import json
import os
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

from speck.stack_edu import _fetch_swh_blob
from speck.stack_v3 import _secret_counts
from speck.stack_v3_refine import _english_prose_result, _sample_partition
from speck.web_sample import (
    _digest,
    _dump_line,
    _exact_keys,
    _fingerprint,
    _integer,
    _path,
    _raw_pii,
    _sha256,
    _strings,
    _write_json,
)

FORMAT = "speck_megamath_code_sample"
FORMAT_VERSION = 1
REPORT_FORMAT = "speck_megamath_code_sample_result"


def validate_megamath_code_config(config, *, config_dir=None):
    """Validate the immutable metadata, filtering, fetching, and partition contract."""

    config_dir = Path(config_dir or ".").resolve()
    _exact_keys(
        config,
        {
            "format",
            "format_version",
            "status",
            "source",
            "rights",
            "filters",
            "fetch",
            "downstream_partition",
            "output_directory",
        },
        "MegaMath code sample",
    )
    if config["format"] != FORMAT or config["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported MegaMath code sample format")
    if config["status"] != "bounded_sample_authorized_not_training_authority":
        raise ValueError("MegaMath code sample must remain non-authoritative")

    source = config["source"]
    _exact_keys(
        source,
        {"id", "repo", "revision", "official_url", "path", "sha256", "rows"},
        "source",
    )
    for key in ("id", "repo", "revision", "official_url"):
        if not isinstance(source[key], str) or not source[key]:
            raise ValueError(f"source.{key} must be non-empty")
    if Path(source["id"]).name != source["id"]:
        raise ValueError("source.id must be a path component")
    normalized_source = {
        **source,
        "path": _path(source["path"], "source.path", config_dir),
        "sha256": _digest(source["sha256"], "source.sha256"),
        "rows": _integer(source["rows"], "source.rows", 1),
    }

    rights = config["rights"]
    _exact_keys(
        rights,
        {"dataset_license", "upstream_terms", "authority", "redistribution"},
        "rights",
    )
    if rights["authority"] != "manual_review_required" or any(
        not isinstance(value, str) or not value for value in rights.values()
    ):
        raise ValueError("MegaMath code rights must remain a manual-review gate")

    filters = config["filters"]
    _exact_keys(
        filters,
        {
            "license_type",
            "accepted_detected_licenses",
            "accepted_encodings",
            "min_file_bytes",
            "max_file_bytes",
            "maximum_bytes_per_repository",
            "allowed_email_placeholders",
            "allowed_ipv4_placeholders",
            "English_prose",
        },
        "filters",
    )
    if filters["license_type"] != "permissive":
        raise ValueError("MegaMath code requires the permissive source subset")
    minimum = _integer(filters["min_file_bytes"], "min_file_bytes", 1)
    maximum = _integer(filters["max_file_bytes"], "max_file_bytes", minimum)
    english = filters["English_prose"]
    _exact_keys(
        english,
        {"detector", "minimum_alphabetic_characters", "minimum_probability"},
        "English_prose",
    )
    probability = english["minimum_probability"]
    if (
        english["detector"] != "py3langid==0.3.0"
        or isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not 0 < probability <= 1
    ):
        raise ValueError("invalid MegaMath code English-prose policy")
    normalized_filters = {
        **filters,
        "accepted_detected_licenses": _strings(
            filters["accepted_detected_licenses"], "accepted_detected_licenses"
        ),
        "accepted_encodings": _strings(filters["accepted_encodings"], "accepted_encodings"),
        "min_file_bytes": minimum,
        "max_file_bytes": maximum,
        "maximum_bytes_per_repository": _integer(
            filters["maximum_bytes_per_repository"], "maximum_bytes_per_repository", 1
        ),
        "allowed_email_placeholders": _strings(
            filters["allowed_email_placeholders"], "allowed_email_placeholders", allow_empty=True
        ),
        "allowed_ipv4_placeholders": _strings(
            filters["allowed_ipv4_placeholders"], "allowed_ipv4_placeholders", allow_empty=True
        ),
        "English_prose": {
            **english,
            "minimum_alphabetic_characters": _integer(
                english["minimum_alphabetic_characters"],
                "minimum_alphabetic_characters",
                1,
            ),
            "minimum_probability": float(probability),
        },
    }

    fetch = config["fetch"]
    _exact_keys(fetch, {"base_url", "workers", "timeout_seconds", "attempts"}, "fetch")
    if fetch["base_url"] != "https://softwareheritage.s3.amazonaws.com/content/":
        raise ValueError("MegaMath code blobs must use Software Heritage S3")
    normalized_fetch = {
        **fetch,
        "workers": _integer(fetch["workers"], "fetch.workers", 1),
        "timeout_seconds": _integer(fetch["timeout_seconds"], "fetch.timeout_seconds", 1),
        "attempts": _integer(fetch["attempts"], "fetch.attempts", 1),
    }

    partition = config["downstream_partition"]
    _exact_keys(
        partition,
        {
            "seed",
            "category",
            "modulus",
            "evaluation_remainders",
            "training_bytes",
            "evaluation_bytes",
        },
        "downstream_partition",
    )
    modulus = _integer(partition["modulus"], "partition.modulus", 2)
    remainders = partition["evaluation_remainders"]
    if (
        partition["category"] != "math"
        or not isinstance(remainders, list)
        or not remainders
        or len(remainders) != len(set(remainders))
        or any(
            isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < modulus
            for value in remainders
        )
        or len(remainders) == modulus
    ):
        raise ValueError("invalid MegaMath code partition")
    normalized_partition = {
        **partition,
        "seed": _integer(partition["seed"], "partition.seed"),
        "modulus": modulus,
        "evaluation_remainders": sorted(remainders),
        "training_bytes": _integer(partition["training_bytes"], "training_bytes", 1),
        "evaluation_bytes": _integer(partition["evaluation_bytes"], "evaluation_bytes", 1),
    }
    normalized = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "status": config["status"],
        "source": normalized_source,
        "rights": rights,
        "filters": normalized_filters,
        "fetch": normalized_fetch,
        "downstream_partition": normalized_partition,
        "output_directory": _path(config["output_directory"], "output_directory", config_dir),
    }
    normalized["plan_fingerprint"] = _fingerprint(normalized)
    return normalized


def load_megamath_code_config(path):
    path = Path(path).resolve()
    return validate_megamath_code_config(json.loads(path.read_text()), config_dir=path.parent)


def _metadata_candidates(source, filters, seed, counts):
    parquet = pq.ParquetFile(source["path"])
    if parquet.metadata.num_rows != source["rows"] or set(parquet.schema_arrow.names) != {
        "file_info",
        "repo_info",
    }:
        raise ValueError("MegaMath code metadata schema or row count mismatch")
    accepted_licenses = set(filters["accepted_detected_licenses"])
    accepted_encodings = set(filters["accepted_encodings"])
    candidates = []
    seen = set()
    for batch in parquet.iter_batches(batch_size=8192, use_threads=False):
        for row in batch.to_pylist():
            counts["metadata_rows_seen"] += 1
            file_info = row.get("file_info") or {}
            repo_info = row.get("repo_info") or {}
            blob_id = file_info.get("blob_id")
            licenses = file_info.get("detected_licenses") or []
            if (
                not isinstance(blob_id, str)
                or len(blob_id) != 40
                or any(character not in "0123456789abcdef" for character in blob_id)
                or blob_id in seen
            ):
                counts["metadata_identity_rejected"] += 1
                continue
            seen.add(blob_id)
            if file_info.get("license_type") != filters["license_type"]:
                counts["license_type_rejected"] += 1
                continue
            if not licenses or any(value not in accepted_licenses for value in licenses):
                counts["license_allowlist_rejected"] += 1
                continue
            if file_info.get("src_encoding") not in accepted_encodings:
                counts["encoding_rejected"] += 1
                continue
            if (
                not filters["min_file_bytes"]
                <= file_info.get("length_bytes", 0)
                <= filters["max_file_bytes"]
            ):
                counts["size_rejected"] += 1
                continue
            if not isinstance(repo_info.get("repo_name"), str) or not repo_info["repo_name"]:
                counts["repository_rejected"] += 1
                continue
            priority = hashlib.sha256(f"{seed}\0{source['id']}\0{blob_id}".encode()).digest()
            candidates.append((priority, file_info, repo_info))
    candidates.sort(key=lambda item: item[0])
    return candidates


def sample_megamath_code(config, *, fetch_blob=None, restart=False):
    """Fetch deterministic metadata candidates until both math partitions pass."""

    if "plan_fingerprint" not in config:
        config = validate_megamath_code_config(config)
    else:
        payload = {key: value for key, value in config.items() if key != "plan_fingerprint"}
        if config["plan_fingerprint"] != _fingerprint(payload):
            raise ValueError("normalized MegaMath code fingerprint mismatch")
    source = config["source"]
    path = Path(source["path"])
    if not path.is_file() or _sha256(path) != source["sha256"]:
        raise ValueError("MegaMath code metadata identity mismatch")
    filters = config["filters"]
    settings = config["downstream_partition"]
    counts = Counter()
    candidates = _metadata_candidates(source, filters, settings["seed"], counts)

    output = Path(config["output_directory"])
    staging = output.with_name(output.name + ".building")
    if output.exists():
        raise FileExistsError(f"MegaMath code sample already exists: {output}")
    if staging.exists():
        if not restart:
            raise FileExistsError(f"incomplete MegaMath code sample exists: {staging}")
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    tokenizer_path = staging / "tokenizer-input.jsonl"
    attribution_path = staging / "attribution.jsonl"
    targets = {"train": settings["training_bytes"], "eval": settings["evaluation_bytes"]}
    partitions = Counter()
    per_repo = Counter()
    repositories = set()
    licenses = Counter()
    fetch_blob = fetch_blob or (lambda blob: _fetch_swh_blob(blob, config["fetch"]))
    allowed_emails = {value.lower() for value in filters["allowed_email_placeholders"]}
    allowed_ips = set(filters["allowed_ipv4_placeholders"])

    with (
        tokenizer_path.open("w", encoding="utf-8") as tokenizer,
        attribution_path.open("w", encoding="utf-8") as attribution,
        ThreadPoolExecutor(max_workers=config["fetch"]["workers"]) as executor,
    ):
        chunk_size = config["fetch"]["workers"] * 2
        for start in range(0, len(candidates), chunk_size):
            if all(partitions[f"{split}_bytes"] >= target for split, target in targets.items()):
                break
            chunk = candidates[start : start + chunk_size]
            blobs = [file_info["blob_id"] for _, file_info, _ in chunk]
            for (_, file_info, repo_info), (status, raw) in zip(
                chunk, executor.map(fetch_blob, blobs)
            ):
                counts[f"blob_fetch_{status}"] += 1
                if status != "ok" or raw is None:
                    continue
                blob_id = file_info["blob_id"]
                if hashlib.sha1(raw).hexdigest() != blob_id:
                    counts["content_hash_rejected"] += 1
                    continue
                if len(raw) != file_info["length_bytes"]:
                    counts["content_length_metadata_mismatch"] += 1
                try:
                    text = raw.decode(file_info["src_encoding"], errors="strict")
                except (LookupError, UnicodeDecodeError):
                    counts["decode_rejected"] += 1
                    continue
                emails, ips = _raw_pii(text, allowed_emails, allowed_ips)
                if emails or ips:
                    counts["raw_PII_rejected"] += 1
                    continue
                if sum(_secret_counts(text).values()):
                    counts["high_confidence_secret_rejected"] += 1
                    continue
                prose, probability = _english_prose_result(
                    text, file_info["language"], filters["English_prose"]
                )
                counts[f"prose_{prose}"] += 1
                if prose == "non_English":
                    counts["non_English_prose_rejected"] += 1
                    continue
                size = len(raw)
                repository = repo_info["repo_name"]
                if per_repo[repository] + size > filters["maximum_bytes_per_repository"]:
                    counts["repository_cap_rejected"] += 1
                    continue
                record = {
                    "text": text,
                    "source": source["id"],
                    "content_id": file_info["content_id"],
                    "released_content_sha256": hashlib.sha256(raw).hexdigest(),
                    "repo_path": repository,
                    "repo_id": None,
                    "commit_id": None,
                    "file_path": file_info["path"],
                    "language": file_info["language"],
                    "detected_licenses": file_info["detected_licenses"],
                    "rights_status": "original_file_license_and_attribution_manual_review_required",
                    "url": repo_info.get("repo_url"),
                    "host": "github.com"
                    if str(repo_info.get("repo_url", "")).startswith("https://github.com/")
                    else None,
                    "size_bytes": size,
                    "declared_length_bytes": file_info["length_bytes"],
                    "src_encoding": file_info["src_encoding"],
                    "detected_English_probability": probability,
                }
                partition = _sample_partition(record, settings)
                if partitions[f"{partition}_bytes"] >= targets[partition]:
                    counts[f"{partition}_quota_already_filled"] += 1
                    continue
                _dump_line(tokenizer, record)
                _dump_line(
                    attribution, {key: value for key, value in record.items() if key != "text"}
                )
                partitions[f"{partition}_bytes"] += size
                partitions[f"{partition}_records"] += 1
                per_repo[repository] += size
                repositories.add(repository)
                licenses.update(file_info["detected_licenses"])
                counts["records_sampled"] += 1
        for handle in (tokenizer, attribution):
            handle.flush()
            os.fsync(handle.fileno())
    missing = {
        split: {"bytes": partitions[f"{split}_bytes"], "target_bytes": target}
        for split, target in targets.items()
        if partitions[f"{split}_bytes"] < target
    }
    report = {
        "format": REPORT_FORMAT,
        "format_version": FORMAT_VERSION,
        "status": (
            "bounded_sample_incomplete_not_training_authority"
            if missing
            else "bounded_sample_complete_not_training_authority"
        ),
        "plan_fingerprint": config["plan_fingerprint"],
        "source": source,
        "rights": config["rights"],
        "filters": filters,
        "fetch": config["fetch"],
        "counts": dict(sorted(counts.items())),
        "metadata_candidates": len(candidates),
        "repositories": len(repositories),
        "accepted_licenses": dict(licenses.most_common()),
        "downstream_partition": {**settings, "observed": dict(sorted(partitions.items()))},
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
            "metadata_identity_schema_and_rows": "pass",
            "SWH_blob_identity": "pass_for_sampled_records",
            "engineering_license_allowlist": "pass",
            "English_comments_and_docstrings": "pass_for_declared_extractors",
            "conservative_PII_and_secret_filter": "pass",
            "downstream_tokenizer_partition_yield": "fail" if missing else "pass",
            "dedicated_secret_scanner": "pending",
            "benchmark_contamination": "pending_evaluation_firewall",
            "cross_source_near_duplicates": "pending",
            "manual_legal_acceptance": "pending",
            "training_authority": "blocked",
        },
    }
    if missing:
        report["missing_downstream_partition_bytes"] = missing
    _write_json(staging / "report.json", report)
    if missing:
        raise RuntimeError(f"MegaMath code cannot fill tokenizer partitions: {missing}")
    os.replace(staging, output)
    return report
