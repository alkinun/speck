"""Build a bounded Python-Edu sample from pinned metadata and SWH blobs."""

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

FORMAT = "speck_python_edu_sample"
FORMAT_VERSION = 1
REPORT_FORMAT = "speck_python_edu_sample_result"


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


def _exact_keys(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{name} must contain exactly: {', '.join(sorted(keys))}")


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


def validate_python_edu_config(config, *, config_dir=None):
    """Validate and normalize a bounded Python-Edu acquisition plan."""

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
        "Python-Edu sample",
    )
    if config["format"] != FORMAT or config["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported Python-Edu sample format")
    if config["status"] != "bounded_sample_authorized_not_training_authority":
        raise ValueError("Python-Edu sample must remain non-authoritative")
    source = config["source"]
    _exact_keys(
        source,
        {"repo", "revision", "official_url", "path", "sha256", "rows"},
        "source",
    )
    normalized_source = {
        **source,
        "path": _path(source["path"], "source.path", config_dir),
        "sha256": _digest(source["sha256"], "source.sha256"),
        "rows": _integer(source["rows"], "source.rows", 1),
    }
    for key in ("repo", "revision", "official_url"):
        if not isinstance(source[key], str) or not source[key]:
            raise ValueError(f"source.{key} must be non-empty")
    rights = config["rights"]
    _exact_keys(
        rights,
        {"dataset_license", "source_lineage", "file_license_metadata", "authority"},
        "rights",
    )
    if rights["file_license_metadata"] != "absent" or rights["authority"] != "manual_review_required":
        raise ValueError("Python-Edu missing file licenses must remain a manual gate")
    if any(not isinstance(rights[key], str) or not rights[key] for key in rights):
        raise ValueError("Python-Edu rights fields must be non-empty strings")

    filters = config["filters"]
    _exact_keys(
        filters,
        {
            "minimum_integer_score",
            "min_file_bytes",
            "max_file_bytes",
            "max_bytes_per_repository",
            "sample_bytes",
            "English_prose",
        },
        "filters",
    )
    minimum = _integer(filters["min_file_bytes"], "min_file_bytes", 1)
    maximum = _integer(filters["max_file_bytes"], "max_file_bytes", minimum)
    english = filters["English_prose"]
    _exact_keys(
        english,
        {"detector", "minimum_alphabetic_characters", "minimum_probability"},
        "English_prose",
    )
    if english["detector"] != "py3langid==0.3.0":
        raise ValueError("Python-Edu English detector must be pinned py3langid")
    probability = english["minimum_probability"]
    if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not 0 < probability <= 1:
        raise ValueError("minimum English probability must be in (0, 1]")

    fetch = config["fetch"]
    _exact_keys(fetch, {"base_url", "workers", "timeout_seconds", "attempts"}, "fetch")
    if fetch["base_url"] != "https://softwareheritage.s3.amazonaws.com/content/":
        raise ValueError("Python-Edu blobs must use the declared SWH source")
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
        partition["category"] != "code"
        or not isinstance(remainders, list)
        or not remainders
        or len(remainders) != len(set(remainders))
        or any(not isinstance(x, int) or not 0 <= x < modulus for x in remainders)
        or len(remainders) == modulus
    ):
        raise ValueError("invalid Python-Edu downstream partition")
    normalized = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "status": config["status"],
        "source": normalized_source,
        "rights": rights,
        "filters": {
            **filters,
            "minimum_integer_score": _integer(
                filters["minimum_integer_score"], "minimum_integer_score", 1
            ),
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
        "fetch": {
            **fetch,
            "workers": _integer(fetch["workers"], "fetch.workers", 1),
            "timeout_seconds": _integer(fetch["timeout_seconds"], "fetch.timeout_seconds", 1),
            "attempts": _integer(fetch["attempts"], "fetch.attempts", 1),
        },
        "downstream_partition": {
            **partition,
            "seed": _integer(partition["seed"], "partition.seed"),
            "modulus": modulus,
            "evaluation_remainders": sorted(remainders),
            "training_bytes": _integer(partition["training_bytes"], "training_bytes", 1),
            "evaluation_bytes": _integer(partition["evaluation_bytes"], "evaluation_bytes", 1),
        },
        "output_directory": _path(config["output_directory"], "output_directory", config_dir),
    }
    normalized["plan_fingerprint"] = _fingerprint(normalized)
    return normalized


def load_python_edu_config(path):
    path = Path(path).resolve()
    return validate_python_edu_config(json.loads(path.read_text()), config_dir=path.parent)


def _metadata_rows(path):
    parquet = pq.ParquetFile(path)
    required = {"blob_id", "repo_name", "path", "length_bytes", "score", "int_score"}
    if set(parquet.schema_arrow.names) != required:
        raise ValueError("Python-Edu metadata schema mismatch")
    for batch in parquet.iter_batches(columns=sorted(required), batch_size=4096, use_threads=False):
        yield from batch.to_pylist()


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _dump_line(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def sample_python_edu(config, *, fetch_blob=None, restart=False):
    """Fetch and filter a bounded score-4+ Python-Edu sample."""

    if "plan_fingerprint" not in config:
        config = validate_python_edu_config(config)
    else:
        payload = {key: value for key, value in config.items() if key != "plan_fingerprint"}
        if config["plan_fingerprint"] != _fingerprint(payload):
            raise ValueError("normalized Python-Edu plan fingerprint mismatch")
    source_path = Path(config["source"]["path"])
    if (
        not source_path.is_file()
        or _sha256(source_path) != config["source"]["sha256"]
        or pq.ParquetFile(source_path).metadata.num_rows != config["source"]["rows"]
    ):
        raise ValueError("Python-Edu metadata identity mismatch")
    output = Path(config["output_directory"])
    staging = output.with_name(output.name + ".building")
    if output.exists():
        raise FileExistsError(f"Python-Edu sample already exists: {output}")
    if staging.exists():
        if not restart:
            raise FileExistsError(f"incomplete Python-Edu sample exists: {staging}")
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    tokenizer_path = staging / "tokenizer-input.jsonl"
    attribution_path = staging / "attribution.jsonl"
    counts = Counter()
    partitions = Counter()
    per_repo = Counter()
    seen_blobs = set()
    sampled_bytes = 0
    fetch_blob = fetch_blob or (lambda blob_id: _fetch_swh_blob(blob_id, config["fetch"]))
    with (
        tokenizer_path.open("w", encoding="utf-8") as tokenizer,
        attribution_path.open("w", encoding="utf-8") as attribution,
        ThreadPoolExecutor(max_workers=config["fetch"]["workers"]) as executor,
    ):
        pending = []

        def consume_pending():
            nonlocal sampled_bytes
            for row, (status, raw) in zip(
                pending,
                executor.map(fetch_blob, [item["blob_id"] for item in pending]),
            ):
                counts[f"blob_fetch_{status}"] += 1
                if sampled_bytes >= config["filters"]["sample_bytes"]:
                    continue
                if status != "ok" or raw is None:
                    continue
                if hashlib.sha1(raw).hexdigest() != row["blob_id"]:
                    counts["content_hash_rejected"] += 1
                    continue
                if len(raw) != row["length_bytes"]:
                    counts["content_length_metadata_mismatch"] += 1
                try:
                    text = raw.decode("utf-8", errors="strict")
                except UnicodeDecodeError:
                    counts["decode_rejected"] += 1
                    continue
                if sum(_secret_counts(text).values()):
                    counts["high_confidence_secret_rejected"] += 1
                    continue
                prose, _ = _english_prose_result(
                    text, "Python", config["filters"]["English_prose"]
                )
                counts[f"prose_{prose}"] += 1
                if prose == "non_English":
                    counts["non_English_prose_rejected"] += 1
                    continue
                size = len(raw)
                repo = row["repo_name"]
                if per_repo[repo] + size > config["filters"]["max_bytes_per_repository"]:
                    counts["repository_cap_rejected"] += 1
                    continue
                record = {
                    "text": text,
                    "source": "python_edu",
                    "content_id": row["blob_id"],
                    "released_content_sha256": hashlib.sha256(raw).hexdigest(),
                    "repo_path": repo,
                    "repo_id": None,
                    "commit_id": None,
                    "file_path": row["path"],
                    "language": "Python",
                    "detected_licenses": [],
                    "rights_status": "file_license_metadata_absent_manual_review_required",
                    "size_bytes": size,
                    "declared_length_bytes": row["length_bytes"],
                    "score": row["score"],
                    "int_score": row["int_score"],
                }
                partition = _sample_partition(record, config["downstream_partition"])
                partitions[f"{partition}_bytes"] += size
                partitions[f"{partition}_records"] += 1
                _dump_line(tokenizer, record)
                _dump_line(attribution, {key: item for key, item in record.items() if key != "text"})
                sampled_bytes += size
                per_repo[repo] += size
                counts["records_sampled"] += 1

        for row in _metadata_rows(source_path):
            if sampled_bytes >= config["filters"]["sample_bytes"]:
                break
            counts["metadata_rows_seen"] += 1
            blob_id = row.get("blob_id")
            if not isinstance(blob_id, str) or len(blob_id) != 40 or blob_id in seen_blobs:
                counts["metadata_identity_rejected"] += 1
                continue
            if row.get("int_score", 0) < config["filters"]["minimum_integer_score"]:
                counts["score_rejected"] += 1
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
        if pending and sampled_bytes < config["filters"]["sample_bytes"]:
            consume_pending()
        for handle in (tokenizer, attribution):
            handle.flush()
            os.fsync(handle.fileno())

    settings = config["downstream_partition"]
    missing = {
        split: {"bytes": partitions[split], "target_bytes": target}
        for split, target in {
            "train_bytes": settings["training_bytes"],
            "eval_bytes": settings["evaluation_bytes"],
        }.items()
        if partitions[split] < target
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
        "rights": config["rights"],
        "filters": config["filters"],
        "fetch": config["fetch"],
        "counts": dict(sorted(counts.items())),
        "sampled_bytes": sampled_bytes,
        "repositories": len(per_repo),
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
            "metadata_identity_and_schema": "pass",
            "SWH_content_identity": "pass_for_sampled_records",
            "score_size_repository_English_and_secret_filters": "pass",
            "downstream_tokenizer_partition_yield": "fail" if missing else "pass",
            "file_level_license_evidence": "blocked_metadata_absent",
            "dedicated_secret_scanner": "pending",
            "benchmark_contamination": "pending",
            "cross_source_near_duplicates": "pending",
            "manual_legal_acceptance": "pending",
            "training_authority": "blocked",
        },
    }
    if missing:
        report["missing_downstream_partition_bytes"] = missing
    _write_json(staging / "report.json", report)
    if missing:
        raise RuntimeError(f"Python-Edu sample cannot fill tokenizer partitions: {missing}")
    os.replace(staging, output)
    return report
