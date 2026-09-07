"""Build a bounded, provenance-preserving Stack-Edu sample from SWH blobs."""

import gzip
import hashlib
import json
import os
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
import requests

from speck.stack_v3 import _secret_counts
from speck.stack_v3_refine import _english_prose_result

FORMAT = "speck_stack_edu_sample"
FORMAT_VERSION = 1
REPORT_FORMAT = "speck_stack_edu_sample_result"


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


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _digest(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _path(value, name, config_dir):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(value).expanduser()
    return str((config_dir / path).resolve() if not path.is_absolute() else path.resolve())


def _exact_keys(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{name} must contain exactly: {', '.join(sorted(keys))}")


def validate_stack_edu_config(config, *, config_dir=None):
    """Validate a bounded Stack-Edu metadata and blob-fetch plan."""

    config_dir = Path(config_dir or ".").resolve()
    _exact_keys(
        config,
        {"format", "format_version", "status", "source", "filters", "fetch", "output_directory"},
        "Stack-Edu sample",
    )
    if config["format"] != FORMAT or config["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported Stack-Edu sample format")
    if config["status"] != "bounded_sample_authorized_not_training_authority":
        raise ValueError("Stack-Edu sample must remain non-authoritative")
    source = config["source"]
    _exact_keys(source, {"repo", "revision", "official_url", "metadata_files"}, "source")
    for key in ("repo", "revision", "official_url"):
        if not isinstance(source[key], str) or not source[key]:
            raise ValueError(f"source.{key} must be non-empty")
    files = source["metadata_files"]
    if not isinstance(files, list) or not files:
        raise ValueError("source.metadata_files must be non-empty")
    normalized_files = []
    languages = []
    for index, item in enumerate(files):
        _exact_keys(item, {"language", "path", "sha256", "rows"}, f"metadata file {index}")
        language = item["language"]
        if not isinstance(language, str) or not language:
            raise ValueError("metadata language must be non-empty")
        languages.append(language)
        normalized_files.append(
            {
                **item,
                "path": _path(item["path"], f"metadata {language} path", config_dir),
                "sha256": _digest(item["sha256"], f"metadata {language} sha256"),
                "rows": _integer(item["rows"], f"metadata {language} rows", 1),
            }
        )
    if len(languages) != len(set(languages)):
        raise ValueError("metadata languages must be unique")

    filters = config["filters"]
    _exact_keys(
        filters,
        {
            "minimum_integer_score",
            "license_type",
            "accepted_detected_licenses",
            "accepted_encodings",
            "min_file_bytes",
            "max_file_bytes",
            "max_bytes_per_language_per_repository",
            "language_sample_bytes",
            "English_prose",
        },
        "filters",
    )
    if filters["license_type"] != "permissive":
        raise ValueError("Stack-Edu only permits the permissive subset")
    minimum_score = _integer(filters["minimum_integer_score"], "minimum_integer_score", 1)
    licenses = filters["accepted_detected_licenses"]
    encodings = filters["accepted_encodings"]
    for values, name in ((licenses, "licenses"), (encodings, "encodings")):
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value for value in values)
            or len(values) != len(set(values))
        ):
            raise ValueError(f"accepted {name} must be unique non-empty strings")
    minimum = _integer(filters["min_file_bytes"], "min_file_bytes", 1)
    maximum = _integer(filters["max_file_bytes"], "max_file_bytes", 1)
    if minimum > maximum:
        raise ValueError("minimum file bytes cannot exceed maximum")
    repo_cap = _integer(
        filters["max_bytes_per_language_per_repository"], "repository byte cap", 1
    )
    targets = filters["language_sample_bytes"]
    if not isinstance(targets, dict) or set(targets) != set(languages):
        raise ValueError("language sample targets must exactly match metadata languages")
    normalized_targets = {
        language: _integer(targets[language], f"language {language} bytes", 1)
        for language in languages
    }
    english = filters["English_prose"]
    _exact_keys(
        english,
        {"detector", "minimum_alphabetic_characters", "minimum_probability"},
        "English_prose",
    )
    if english["detector"] != "py3langid==0.3.0":
        raise ValueError("Stack-Edu English filtering must use pinned py3langid")
    letters = _integer(
        english["minimum_alphabetic_characters"], "minimum alphabetic characters", 1
    )
    probability = english["minimum_probability"]
    if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not 0 < probability <= 1:
        raise ValueError("minimum English probability must be in (0, 1]")

    fetch = config["fetch"]
    _exact_keys(fetch, {"base_url", "workers", "timeout_seconds", "attempts"}, "fetch")
    if fetch["base_url"] != "https://softwareheritage.s3.amazonaws.com/content/":
        raise ValueError("Stack-Edu blobs must use the declared Software Heritage S3 source")
    normalized_fetch = {
        **fetch,
        "workers": _integer(fetch["workers"], "fetch.workers", 1),
        "timeout_seconds": _integer(fetch["timeout_seconds"], "fetch.timeout_seconds", 1),
        "attempts": _integer(fetch["attempts"], "fetch.attempts", 1),
    }
    normalized = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "status": config["status"],
        "source": {**source, "metadata_files": normalized_files},
        "filters": {
            **filters,
            "minimum_integer_score": minimum_score,
            "accepted_detected_licenses": sorted(licenses),
            "accepted_encodings": sorted(encodings),
            "min_file_bytes": minimum,
            "max_file_bytes": maximum,
            "max_bytes_per_language_per_repository": repo_cap,
            "language_sample_bytes": normalized_targets,
            "English_prose": {
                **english,
                "minimum_alphabetic_characters": letters,
                "minimum_probability": float(probability),
            },
        },
        "fetch": normalized_fetch,
        "output_directory": _path(config["output_directory"], "output_directory", config_dir),
    }
    normalized["plan_fingerprint"] = _fingerprint(normalized)
    return normalized


def load_stack_edu_config(path):
    path = Path(path).resolve()
    return validate_stack_edu_config(
        json.loads(path.read_text(encoding="utf-8")), config_dir=path.parent
    )


def _validated_config(config):
    if "plan_fingerprint" not in config:
        return validate_stack_edu_config(config)
    payload = {key: value for key, value in config.items() if key != "plan_fingerprint"}
    if config["plan_fingerprint"] != _fingerprint(payload):
        raise ValueError("normalized Stack-Edu plan fingerprint mismatch")
    return config


def _fetch_swh_blob(blob_id, settings):
    url = settings["base_url"] + blob_id
    for attempt in range(settings["attempts"]):
        try:
            response = requests.get(url, timeout=settings["timeout_seconds"])
            if response.status_code == 404:
                return "missing", None
            response.raise_for_status()
            return "ok", gzip.decompress(response.content)
        except (OSError, requests.RequestException, gzip.BadGzipFile):
            if attempt + 1 == settings["attempts"]:
                return "error", None
            time.sleep(min(2**attempt, 4))
    return "error", None


def _metadata_rows(path):
    parquet = pq.ParquetFile(path)
    required = {
        "blob_id",
        "language",
        "repo_name",
        "path",
        "src_encoding",
        "length_bytes",
        "score",
        "int_score",
        "detected_licenses",
        "license_type",
    }
    if required - set(parquet.schema_arrow.names):
        raise ValueError(f"Stack-Edu metadata schema mismatch: {path}")
    for batch in parquet.iter_batches(columns=sorted(required), batch_size=2048, use_threads=False):
        yield from batch.to_pylist()


def _dump_line(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def sample_stack_edu(config, *, fetch_blob=None, restart=False):
    """Fetch a bounded high-score Stack-Edu sample with immutable provenance."""

    config = _validated_config(config)
    for item in config["source"]["metadata_files"]:
        path = Path(item["path"])
        if (
            not path.is_file()
            or _sha256(path) != item["sha256"]
            or pq.ParquetFile(path).metadata.num_rows != item["rows"]
        ):
            raise ValueError(f"Stack-Edu metadata identity mismatch: {path}")
    output = Path(config["output_directory"])
    staging = output.with_name(output.name + ".building")
    if output.exists():
        raise FileExistsError(f"Stack-Edu sample already exists: {output}")
    if staging.exists():
        if not restart:
            raise FileExistsError(f"incomplete Stack-Edu sample exists: {staging}")
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    tokenizer_path = staging / "tokenizer-input.jsonl"
    attribution_path = staging / "attribution.jsonl"
    targets = config["filters"]["language_sample_bytes"]
    sampled = Counter()
    counts = Counter()
    licenses = Counter()
    repositories = set()
    seen_blobs = set()
    per_repo = Counter()
    fetch_blob = fetch_blob or (lambda blob_id: _fetch_swh_blob(blob_id, config["fetch"]))

    with (
        tokenizer_path.open("w", encoding="utf-8") as tokenizer_output,
        attribution_path.open("w", encoding="utf-8") as attribution_output,
        ThreadPoolExecutor(max_workers=config["fetch"]["workers"]) as executor,
    ):
        for item in config["source"]["metadata_files"]:
            language = item["language"]
            pending = []

            def consume_pending():
                for row, (status, raw) in zip(pending, executor.map(fetch_blob, [x["blob_id"] for x in pending])):
                    counts[f"blob_fetch_{status}"] += 1
                    if status != "ok" or raw is None:
                        continue
                    blob_id = row["blob_id"]
                    if hashlib.sha1(raw).hexdigest() != blob_id:
                        counts["content_hash_rejected"] += 1
                        continue
                    if len(raw) != row["length_bytes"]:
                        counts["content_length_metadata_mismatch"] += 1
                    try:
                        text = raw.decode(row["src_encoding"], errors="strict")
                    except (LookupError, UnicodeDecodeError):
                        counts["decode_rejected"] += 1
                        continue
                    secret_hits = _secret_counts(text)
                    if sum(secret_hits.values()):
                        counts["high_confidence_secret_rejected"] += 1
                        continue
                    prose_result, _ = _english_prose_result(
                        text, language, config["filters"]["English_prose"]
                    )
                    counts[f"prose_{prose_result}"] += 1
                    if prose_result == "non_English":
                        counts["non_English_prose_rejected"] += 1
                        continue
                    size = len(raw)
                    repo_key = (language, row["repo_name"])
                    if (
                        per_repo[repo_key] + size
                        > config["filters"]["max_bytes_per_language_per_repository"]
                    ):
                        counts["repository_cap_rejected"] += 1
                        continue
                    record = {
                        "text": text,
                        "source": "stack_edu",
                        "content_id": blob_id,
                        "released_content_sha256": hashlib.sha256(raw).hexdigest(),
                        "repo_path": row["repo_name"],
                        "repo_id": None,
                        "commit_id": None,
                        "file_path": row["path"],
                        "language": language,
                        "detected_licenses": row["detected_licenses"],
                        "size_bytes": size,
                        "declared_length_bytes": row["length_bytes"],
                        "score": row["score"],
                        "int_score": row["int_score"],
                        "src_encoding": row["src_encoding"],
                    }
                    _dump_line(tokenizer_output, record)
                    _dump_line(
                        attribution_output,
                        {key: value for key, value in record.items() if key != "text"},
                    )
                    sampled[language] += size
                    per_repo[repo_key] += size
                    repositories.add(row["repo_name"])
                    licenses.update(row["detected_licenses"])
                    counts["records_sampled"] += 1

            for row in _metadata_rows(item["path"]):
                if sampled[language] >= targets[language]:
                    break
                counts["metadata_rows_seen"] += 1
                blob_id = row.get("blob_id")
                detected = row.get("detected_licenses") or []
                if not isinstance(blob_id, str) or len(blob_id) != 40 or blob_id in seen_blobs:
                    counts["metadata_identity_rejected"] += 1
                    continue
                if row.get("language") != language:
                    raise ValueError(f"Stack-Edu metadata language mismatch in {item['path']}")
                if row.get("int_score", 0) < config["filters"]["minimum_integer_score"]:
                    counts["score_rejected"] += 1
                    continue
                if row.get("license_type") != config["filters"]["license_type"]:
                    counts["license_type_rejected"] += 1
                    continue
                if not detected or any(
                    license_id not in config["filters"]["accepted_detected_licenses"]
                    for license_id in detected
                ):
                    counts["license_allowlist_rejected"] += 1
                    continue
                if row.get("src_encoding") not in config["filters"]["accepted_encodings"]:
                    counts["encoding_rejected"] += 1
                    continue
                if not config["filters"]["min_file_bytes"] <= row.get(
                    "length_bytes", 0
                ) <= config["filters"]["max_file_bytes"]:
                    counts["size_rejected"] += 1
                    continue
                seen_blobs.add(blob_id)
                pending.append(row)
                if len(pending) == config["fetch"]["workers"] * 2:
                    consume_pending()
                    pending.clear()
            if pending and sampled[language] < targets[language]:
                consume_pending()
                pending.clear()
        for handle in (tokenizer_output, attribution_output):
            handle.flush()
            os.fsync(handle.fileno())

    missing = {
        language: {"sampled_bytes": sampled[language], "target_bytes": target}
        for language, target in targets.items()
        if sampled[language] < target
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
        "source": config["source"],
        "filters": config["filters"],
        "fetch": config["fetch"],
        "counts": dict(sorted(counts.items())),
        "repositories": len(repositories),
        "sampled_languages": {
            language: {
                "bytes": sampled[language],
                "target_bytes": targets[language],
                "overshoot_bytes": sampled[language] - targets[language],
            }
            for language in targets
        },
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
            "metadata_identity_and_schema": "pass",
            "SWH_content_identity": "pass_for_sampled_records",
            "engineering_license_allowlist": "pass",
            "English_comments": "pass_for_declared_extractors_with_strings_out_of_scope",
            "bounded_language_yield": "fail" if missing else "pass",
            "dedicated_secret_scanner": "pending",
            "benchmark_contamination": "pending",
            "cross_source_near_duplicates": "pending",
            "manual_legal_acceptance": "pending",
            "training_authority": "blocked",
        },
    }
    if missing:
        report["missing_language_quotas"] = missing
    _write_json(staging / "report.json", report)
    if missing:
        raise RuntimeError(f"Stack-Edu metadata exhausted before quotas: {missing}")
    os.replace(staging, output)
    return report
