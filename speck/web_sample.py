"""Build deterministic bounded web samples from pinned Parquet shards."""

import hashlib
import ipaddress
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import pyarrow.parquet as pq

from speck.stack_v3 import _secret_counts
from speck.stack_v3_refine import _language_identifier, _sample_partition

FORMAT = "speck_web_sample"
FORMAT_VERSION = 1
REPORT_FORMAT = "speck_web_sample_result"
_EMAIL = re.compile(r"(?i)(?<![\w.+-])[\w.+-]+@[a-z0-9.-]+\.[a-z]{2,}(?![\w.-])")
_IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")


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


def _strings(values, name, *, allow_empty=False):
    if (
        not isinstance(values, list)
        or (not allow_empty and not values)
        or any(not isinstance(value, str) or not value for value in values)
        or len(values) != len(set(values))
    ):
        raise ValueError(f"{name} must be unique non-empty strings")
    return sorted(values)


def validate_web_sample_config(config, *, config_dir=None):
    """Validate and normalize a single-source bounded web sample plan."""

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
            "downstream_partition",
            "output_directory",
        },
        "web sample",
    )
    if config["format"] != FORMAT or config["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported web sample format")
    if config["status"] != "bounded_sample_authorized_not_training_authority":
        raise ValueError("web sample must remain non-authoritative")
    source = config["source"]
    _exact_keys(
        source,
        {
            "id",
            "repo",
            "revision",
            "official_url",
            "path",
            "sha256",
            "rows",
            "content_column",
            "metadata_json_column",
            "fields",
        },
        "source",
    )
    for key in ("id", "repo", "revision", "official_url", "content_column"):
        if not isinstance(source[key], str) or not source[key]:
            raise ValueError(f"source.{key} must be non-empty")
    if Path(source["id"]).name != source["id"]:
        raise ValueError("source.id must be a path component")
    if source["metadata_json_column"] is not None and not isinstance(
        source["metadata_json_column"], str
    ):
        raise ValueError("metadata_json_column must be null or a string")
    fields = source["fields"]
    _exact_keys(
        fields,
        {"document_id", "url", "language", "language_score", "quality_score"},
        "source.fields",
    )
    for name, value in fields.items():
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"source.fields.{name} must be null or a string")
        if isinstance(value, str) and value.startswith("$metadata.") and not source[
            "metadata_json_column"
        ]:
            raise ValueError(f"source.fields.{name} requires metadata_json_column")
    normalized_source = {
        **source,
        "path": _path(source["path"], "source.path", config_dir),
        "sha256": _digest(source["sha256"], "source.sha256"),
        "rows": _integer(source["rows"], "source.rows", 1),
    }
    rights = config["rights"]
    _exact_keys(rights, {"dataset_license", "upstream_terms", "authority"}, "rights")
    if rights["authority"] != "manual_review_required" or any(
        not isinstance(value, str) or not value for value in rights.values()
    ):
        raise ValueError("web rights must remain a non-empty manual-review gate")

    filters = config["filters"]
    _exact_keys(
        filters,
        {
            "required_language",
            "minimum_language_score",
            "minimum_quality_score",
            "min_document_bytes",
            "max_document_bytes",
            "minimum_alphabetic_ratio",
            "maximum_duplicate_line_ratio",
            "maximum_bytes_per_host",
            "adult_host_terms",
            "allowed_email_placeholders",
            "allowed_ipv4_placeholders",
            "language_detector",
            "minimum_detected_English_probability",
        },
        "filters",
    )
    for key in ("required_language", "language_detector"):
        if not isinstance(filters[key], str) or not filters[key]:
            raise ValueError(f"filters.{key} must be non-empty")
    if filters["language_detector"] != "py3langid==0.3.0":
        raise ValueError("web language detector must be pinned py3langid")
    minimum = _integer(filters["min_document_bytes"], "min_document_bytes", 1)
    maximum = _integer(filters["max_document_bytes"], "max_document_bytes", minimum)
    numeric = {}
    for key in (
        "minimum_language_score",
        "minimum_alphabetic_ratio",
        "maximum_duplicate_line_ratio",
        "minimum_detected_English_probability",
    ):
        value = filters[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"filters.{key} must be in [0, 1]")
        numeric[key] = float(value)
    quality = filters["minimum_quality_score"]
    if quality is not None and (isinstance(quality, bool) or not isinstance(quality, (int, float))):
        raise ValueError("minimum_quality_score must be null or numeric")
    normalized_filters = {
        **filters,
        **numeric,
        "minimum_quality_score": None if quality is None else float(quality),
        "min_document_bytes": minimum,
        "max_document_bytes": maximum,
        "maximum_bytes_per_host": _integer(
            filters["maximum_bytes_per_host"], "maximum_bytes_per_host", 1
        ),
        "adult_host_terms": _strings(filters["adult_host_terms"], "adult_host_terms"),
        "allowed_email_placeholders": _strings(
            filters["allowed_email_placeholders"], "allowed_email_placeholders", allow_empty=True
        ),
        "allowed_ipv4_placeholders": _strings(
            filters["allowed_ipv4_placeholders"], "allowed_ipv4_placeholders", allow_empty=True
        ),
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
        partition["category"] != "web"
        or not isinstance(remainders, list)
        or not remainders
        or len(remainders) != len(set(remainders))
        or any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < modulus for value in remainders)
        or len(remainders) == modulus
    ):
        raise ValueError("invalid web downstream partition")
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
        "downstream_partition": normalized_partition,
        "output_directory": _path(config["output_directory"], "output_directory", config_dir),
    }
    normalized["plan_fingerprint"] = _fingerprint(normalized)
    return normalized


def load_web_sample_config(path):
    path = Path(path).resolve()
    return validate_web_sample_config(json.loads(path.read_text()), config_dir=path.parent)


def _field(row, reference, metadata):
    if reference is None:
        return None
    if reference.startswith("$metadata."):
        return metadata.get(reference.removeprefix("$metadata."))
    return row.get(reference)


def _host(url):
    if not isinstance(url, str):
        return None
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    return parsed.hostname.lower().rstrip(".") if parsed.scheme in {"http", "https"} and parsed.hostname else None


def _duplicate_line_ratio(text):
    lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return 0.0
    return 1 - len(set(lines)) / len(lines)


def _raw_pii(text, allowed_emails, allowed_ips):
    emails = {value.lower() for value in _EMAIL.findall(text)} - allowed_emails
    ips = set()
    for value in _IPV4.findall(text):
        try:
            parsed = ipaddress.ip_address(value)
        except ValueError:
            continue
        if str(parsed) not in allowed_ips:
            ips.add(str(parsed))
    return emails, ips


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


def sample_web_source(config, *, restart=False):
    """Select a deterministic bounded web sample with conservative local filters."""

    if "plan_fingerprint" not in config:
        config = validate_web_sample_config(config)
    else:
        payload = {key: value for key, value in config.items() if key != "plan_fingerprint"}
        if config["plan_fingerprint"] != _fingerprint(payload):
            raise ValueError("normalized web sample fingerprint mismatch")
    source_path = Path(config["source"]["path"])
    parquet = pq.ParquetFile(source_path)
    if (
        not source_path.is_file()
        or _sha256(source_path) != config["source"]["sha256"]
        or parquet.metadata.num_rows != config["source"]["rows"]
    ):
        raise ValueError("web source identity mismatch")
    source = config["source"]
    direct_columns = {source["content_column"]}
    if source["metadata_json_column"]:
        direct_columns.add(source["metadata_json_column"])
    direct_columns.update(
        reference
        for reference in source["fields"].values()
        if isinstance(reference, str) and not reference.startswith("$metadata.")
    )
    if direct_columns - set(parquet.schema_arrow.names):
        raise ValueError("web source schema mismatch")
    output = Path(config["output_directory"])
    staging = output.with_name(output.name + ".building")
    if output.exists():
        raise FileExistsError(f"web sample already exists: {output}")
    if staging.exists():
        if not restart:
            raise FileExistsError(f"incomplete web sample exists: {staging}")
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    tokenizer_path = staging / "tokenizer-input.jsonl"
    attribution_path = staging / "attribution.jsonl"
    counts = Counter()
    partitions = Counter()
    host_bytes = Counter()
    seen = set()
    probabilities = []
    settings = config["downstream_partition"]
    filters = config["filters"]
    targets = {"train": settings["training_bytes"], "eval": settings["evaluation_bytes"]}
    allowed_emails = {value.lower() for value in filters["allowed_email_placeholders"]}
    allowed_ips = set(filters["allowed_ipv4_placeholders"])
    row_groups = sorted(
        range(parquet.num_row_groups),
        key=lambda index: hashlib.sha256(
            f"{settings['seed']}\0{source['id']}\0row-group\0{index}".encode()
        ).digest(),
    )
    row_groups_read = 0
    with (
        tokenizer_path.open("w", encoding="utf-8") as tokenizer,
        attribution_path.open("w", encoding="utf-8") as attribution,
    ):
        for row_group in row_groups:
            if all(partitions[f"{split}_bytes"] >= target for split, target in targets.items()):
                break
            row_groups_read += 1
            rows = parquet.read_row_group(
                row_group, columns=sorted(direct_columns), use_threads=False
            ).to_pylist()
            candidates = []
            for row_index, row in enumerate(rows):
                text = row.get(source["content_column"])
                if not isinstance(text, str):
                    counts["content_rejected"] += 1
                    continue
                digest = hashlib.sha256(text.encode()).hexdigest()
                priority = hashlib.sha256(
                    f"{settings['seed']}\0{source['id']}\0{digest}".encode()
                ).digest()
                candidates.append((priority, row_index, digest, row, text))
            for _, _, digest, row, text in sorted(candidates):
                if all(partitions[f"{split}_bytes"] >= target for split, target in targets.items()):
                    break
                counts["records_considered"] += 1
                metadata = {}
                if source["metadata_json_column"]:
                    try:
                        metadata = json.loads(row[source["metadata_json_column"]])
                    except (TypeError, json.JSONDecodeError):
                        counts["metadata_json_rejected"] += 1
                        continue
                fields = {
                    name: _field(row, reference, metadata)
                    for name, reference in source["fields"].items()
                }
                if fields["document_id"] is None:
                    counts["document_id_rejected"] += 1
                    continue
                language_score = fields["language_score"]
                if fields["language"] != filters["required_language"]:
                    counts["metadata_language_rejected"] += 1
                    continue
                if (
                    isinstance(language_score, bool)
                    or not isinstance(language_score, (int, float))
                    or language_score < filters["minimum_language_score"]
                ):
                    counts["metadata_language_score_rejected"] += 1
                    continue
                minimum_quality = filters["minimum_quality_score"]
                quality = fields["quality_score"]
                if minimum_quality is not None and (
                    isinstance(quality, bool)
                    or not isinstance(quality, (int, float))
                    or quality < minimum_quality
                ):
                    counts["quality_score_rejected"] += 1
                    continue
                raw = text.encode()
                size = len(raw)
                if not filters["min_document_bytes"] <= size <= filters["max_document_bytes"]:
                    counts["size_rejected"] += 1
                    continue
                alphabetic_ratio = sum(character.isalpha() for character in text) / max(len(text), 1)
                if alphabetic_ratio < filters["minimum_alphabetic_ratio"]:
                    counts["alphabetic_ratio_rejected"] += 1
                    continue
                if _duplicate_line_ratio(text) > filters["maximum_duplicate_line_ratio"]:
                    counts["duplicate_lines_rejected"] += 1
                    continue
                host = _host(fields["url"])
                if host is None:
                    counts["url_rejected"] += 1
                    continue
                if any(term in host for term in filters["adult_host_terms"]):
                    counts["adult_host_rejected"] += 1
                    continue
                if host_bytes[host] + size > filters["maximum_bytes_per_host"]:
                    counts["host_cap_rejected"] += 1
                    continue
                if digest in seen:
                    counts["exact_duplicate_rejected"] += 1
                    continue
                emails, ips = _raw_pii(text, allowed_emails, allowed_ips)
                if emails:
                    counts["raw_email_rejected"] += 1
                    continue
                if ips:
                    counts["raw_ipv4_rejected"] += 1
                    continue
                if sum(_secret_counts(text).values()):
                    counts["high_confidence_secret_rejected"] += 1
                    continue
                detected, probability = _language_identifier().classify(text)
                probability = float(probability)
                probabilities.append(probability)
                if detected != "en" or probability < filters[
                    "minimum_detected_English_probability"
                ]:
                    counts["detected_non_English_rejected"] += 1
                    continue
                record = {
                    "text": text,
                    "source": source["id"],
                    "content_id": str(fields["document_id"]),
                    "released_content_sha256": digest,
                    "repo_path": host,
                    "repo_id": None,
                    "commit_id": None,
                    "file_path": fields["url"],
                    "language": "English",
                    "detected_licenses": [],
                    "rights_status": "upstream_web_terms_manual_review_required",
                    "url": fields["url"],
                    "host": host,
                    "size_bytes": size,
                    "language_score": float(language_score),
                    "detected_English_probability": probability,
                    "quality_score": None if quality is None else float(quality),
                }
                partition = _sample_partition(record, settings)
                if partitions[f"{partition}_bytes"] >= targets[partition]:
                    counts[f"{partition}_quota_already_filled"] += 1
                    continue
                _dump_line(tokenizer, record)
                _dump_line(attribution, {key: item for key, item in record.items() if key != "text"})
                seen.add(digest)
                host_bytes[host] += size
                partitions[f"{partition}_bytes"] += size
                partitions[f"{partition}_records"] += 1
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
        "source": config["source"],
        "rights": config["rights"],
        "filters": config["filters"],
        "counts": dict(sorted(counts.items())),
        "source_profile": {
            "row_groups_total": parquet.num_row_groups,
            "row_groups_read": row_groups_read,
            "rows_total": parquet.metadata.num_rows,
            "selected_hosts": len(host_bytes),
            "largest_selected_host_bytes": max(host_bytes.values(), default=0),
            "minimum_observed_detected_English_probability": min(probabilities, default=None),
        },
        "downstream_partition": {
            **settings,
            "observed": dict(sorted(partitions.items())),
        },
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
            "source_identity_schema_and_rows": "pass",
            "declared_quality_and_language_metadata": "pass",
            "independent_English_detection": "pass",
            "conservative_PII_secret_adult_host_and_repetition_filters": "pass",
            "exact_selected_document_deduplication": "pass",
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
        raise RuntimeError(f"web source cannot fill tokenizer partitions: {missing}")
    os.replace(staging, output)
    return report
