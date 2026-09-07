"""Build a bounded, provenance-preserving Python PEP tokenizer source."""

import gzip
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from speck.stack_v3 import _secret_counts
from speck.stack_v3_refine import _english_prose_result, _sample_partition

FORMAT = "speck_common_pile_peps_sample"
FORMAT_VERSION = 1
REPORT_FORMAT = "speck_common_pile_peps_sample_result"


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


def validate_peps_config(config, *, config_dir=None):
    """Validate and normalize a frozen bounded PEP sample plan."""

    config_dir = Path(config_dir or ".").resolve()
    _exact_keys(
        config,
        {"format", "format_version", "status", "source", "filters", "output_directory"},
        "PEP sample",
    )
    if config["format"] != FORMAT or config["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported PEP sample format")
    if config["status"] != "bounded_sample_authorized_not_training_authority":
        raise ValueError("PEP sample must remain non-authoritative")
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
            "accepted_metadata_license",
            "min_document_bytes",
            "max_document_bytes",
            "English_prose",
            "downstream_partition",
        },
        "filters",
    )
    english = filters["English_prose"]
    _exact_keys(
        english,
        {"detector", "minimum_alphabetic_characters", "minimum_probability"},
        "English_prose",
    )
    if english["detector"] != "py3langid==0.3.0":
        raise ValueError("PEP English detector must be pinned py3langid")
    partition = filters["downstream_partition"]
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
    ):
        raise ValueError("invalid PEP downstream partition")
    minimum = _integer(filters["min_document_bytes"], "min_document_bytes", 1)
    maximum = _integer(filters["max_document_bytes"], "max_document_bytes", minimum)
    probability = english["minimum_probability"]
    if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not 0 < probability <= 1:
        raise ValueError("minimum English probability must be in (0, 1]")
    if not isinstance(filters["accepted_metadata_license"], str) or not filters[
        "accepted_metadata_license"
    ]:
        raise ValueError("accepted_metadata_license must be non-empty")
    normalized = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "status": config["status"],
        "source": normalized_source,
        "filters": {
            **filters,
            "min_document_bytes": minimum,
            "max_document_bytes": maximum,
            "English_prose": {
                **english,
                "minimum_alphabetic_characters": _integer(
                    english["minimum_alphabetic_characters"],
                    "minimum_alphabetic_characters",
                    1,
                ),
                "minimum_probability": float(probability),
            },
            "downstream_partition": {
                **partition,
                "seed": _integer(partition["seed"], "partition.seed"),
                "modulus": modulus,
                "evaluation_remainders": sorted(remainders),
                "training_bytes": _integer(partition["training_bytes"], "training_bytes", 1),
                "evaluation_bytes": _integer(
                    partition["evaluation_bytes"], "evaluation_bytes", 1
                ),
            },
        },
        "output_directory": _path(config["output_directory"], "output_directory", config_dir),
    }
    normalized["plan_fingerprint"] = _fingerprint(normalized)
    return normalized


def load_peps_config(path):
    path = Path(path).resolve()
    return validate_peps_config(json.loads(path.read_text()), config_dir=path.parent)


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


def sample_common_pile_peps(config, *, restart=False):
    """Filter all pinned PEP rows and preserve deterministic tokenizer partitions."""

    if "plan_fingerprint" not in config:
        config = validate_peps_config(config)
    else:
        payload = {key: value for key, value in config.items() if key != "plan_fingerprint"}
        if config["plan_fingerprint"] != _fingerprint(payload):
            raise ValueError("normalized PEP plan fingerprint mismatch")
    source_path = Path(config["source"]["path"])
    if not source_path.is_file() or _sha256(source_path) != config["source"]["sha256"]:
        raise ValueError("PEP source identity mismatch")
    output = Path(config["output_directory"])
    staging = output.with_name(output.name + ".building")
    if output.exists():
        raise FileExistsError(f"PEP sample already exists: {output}")
    if staging.exists():
        if not restart:
            raise FileExistsError(f"incomplete PEP sample exists: {staging}")
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    tokenizer_path = staging / "tokenizer-input.jsonl"
    attribution_path = staging / "attribution.jsonl"
    counts = Counter()
    partitions = Counter()
    seen = set()
    settings = config["filters"]["downstream_partition"]
    with (
        gzip.open(source_path, "rt", encoding="utf-8") as source,
        tokenizer_path.open("w", encoding="utf-8") as tokenizer,
        attribution_path.open("w", encoding="utf-8") as attribution,
    ):
        for line in source:
            counts["rows_seen"] += 1
            value = json.loads(line)
            metadata = value.get("metadata") or {}
            text = value.get("text")
            if value.get("source") != config["source"]["source_value"] or not isinstance(text, str):
                counts["source_or_content_rejected"] += 1
                continue
            if metadata.get("license") != config["filters"]["accepted_metadata_license"]:
                counts["license_rejected"] += 1
                continue
            size = len(text.encode())
            if not config["filters"]["min_document_bytes"] <= size <= config["filters"][
                "max_document_bytes"
            ]:
                counts["size_rejected"] += 1
                continue
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest in seen:
                counts["exact_duplicate_rejected"] += 1
                continue
            if sum(_secret_counts(text).values()):
                counts["high_confidence_secret_rejected"] += 1
                continue
            prose, _ = _english_prose_result(text, "Python", config["filters"]["English_prose"])
            counts[f"prose_{prose}"] += 1
            if prose == "non_English":
                counts["non_English_rejected"] += 1
                continue
            record = {
                "text": text,
                "source": "common_pile_python_peps",
                "content_id": str(value.get("id")),
                "released_content_sha256": digest,
                "repo_path": "python/peps",
                "repo_id": None,
                "commit_id": None,
                "file_path": f"pep-{metadata.get('pep_number')}.txt",
                "language": "Python",
                "detected_licenses": [metadata.get("license")],
                "size_bytes": size,
                "url": metadata.get("url"),
                "authors": metadata.get("authors"),
                "provenance": metadata.get("provenance"),
            }
            partition = _sample_partition(record, settings)
            partitions[f"{partition}_bytes"] += size
            partitions[f"{partition}_records"] += 1
            _dump_line(tokenizer, record)
            _dump_line(attribution, {key: item for key, item in record.items() if key != "text"})
            seen.add(digest)
            counts["records_sampled"] += 1
        for handle in (tokenizer, attribution):
            handle.flush()
            os.fsync(handle.fileno())
    if counts["rows_seen"] != config["source"]["rows"]:
        raise ValueError("PEP source row count mismatch")
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
        "filters": config["filters"],
        "counts": dict(sorted(counts.items())),
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
            "source_identity_rows_and_metadata_license": "pass",
            "English_prose_and_high_confidence_secret_filter": "pass",
            "exact_document_deduplication": "pass",
            "downstream_tokenizer_partition_yield": "fail" if missing else "pass",
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
        raise RuntimeError(f"PEP sample cannot fill tokenizer partitions: {missing}")
    os.replace(staging, output)
    return report
