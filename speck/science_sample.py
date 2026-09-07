"""Build deterministic, identity-preserving bounded science samples."""

import gzip
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from speck.stack_v3 import _secret_counts
from speck.stack_v3_refine import _language_identifier, _sample_partition
from speck.web_sample import (
    _digest,
    _dump_line,
    _duplicate_line_ratio,
    _exact_keys,
    _fingerprint,
    _host,
    _integer,
    _path,
    _raw_pii,
    _sha256,
    _strings,
    _write_json,
)

FORMAT = "speck_science_sample"
FORMAT_VERSION = 1
REPORT_FORMAT = "speck_science_sample_result"
_SPACED_OCR = re.compile(r"\b(?:[A-Za-z]\s){2,}[A-Za-z]\b")
_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def _nonempty(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be non-empty")
    return value


def _number(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < minimum:
        raise ValueError(f"{name} must be numeric and >= {minimum}")
    return float(value)


def validate_science_sample_config(config, *, config_dir=None):
    """Validate a frozen single-shard science sampling contract."""

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
        "science sample",
    )
    if config["format"] != FORMAT or config["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported science sample format")
    if config["status"] != "bounded_sample_authorized_not_training_authority":
        raise ValueError("science sample must remain non-authoritative")
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
            "file_format",
            "fields",
        },
        "source",
    )
    for key in ("id", "repo", "revision", "official_url"):
        _nonempty(source[key], f"source.{key}")
    if Path(source["id"]).name != source["id"]:
        raise ValueError("source.id must be a path component")
    if source["file_format"] not in {"parquet", "jsonl_gzip", "jsonl_zstd"}:
        raise ValueError("unsupported science source format")
    fields = source["fields"]
    _exact_keys(
        fields,
        {
            "text",
            "document_id",
            "title",
            "url",
            "license",
            "language",
            "language_score",
            "quality_score",
            "date",
            "source_partition",
        },
        "source.fields",
    )
    _nonempty(fields["text"], "source.fields.text")
    for key, value in fields.items():
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"source field {key} must be null or a string")
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
        raise ValueError("science rights must remain a manual-review gate")
    filters = config["filters"]
    _exact_keys(
        filters,
        {
            "required_language",
            "minimum_language_score",
            "minimum_quality_score",
            "language_detector",
            "minimum_detected_English_probability",
            "min_document_bytes",
            "max_document_bytes",
            "minimum_alphabetic_ratio",
            "maximum_duplicate_line_ratio",
            "maximum_spaced_OCR_sequences",
            "maximum_replacement_character_ratio",
            "boilerplate_line_terms",
            "maximum_boilerplate_line_ratio",
            "accepted_document_licenses",
            "minimum_science_term_hits",
            "science_terms",
            "maximum_bytes_per_host",
            "allowed_email_placeholders",
            "allowed_ipv4_placeholders",
        },
        "filters",
    )
    if filters["language_detector"] != "py3langid==0.3.0":
        raise ValueError("science language detector must be pinned py3langid")
    if filters["required_language"] is not None and (
        not isinstance(filters["required_language"], str) or not filters["required_language"]
    ):
        raise ValueError("required_language must be null or a string")
    optional = {}
    for key in ("minimum_language_score", "minimum_quality_score"):
        value = filters[key]
        optional[key] = None if value is None else _number(value, f"filters.{key}")
    ratios = {}
    for key in (
        "minimum_detected_English_probability",
        "minimum_alphabetic_ratio",
        "maximum_duplicate_line_ratio",
        "maximum_replacement_character_ratio",
        "maximum_boilerplate_line_ratio",
    ):
        value = _number(filters[key], f"filters.{key}")
        if value > 1:
            raise ValueError(f"filters.{key} must be in [0, 1]")
        ratios[key] = value
    minimum = _integer(filters["min_document_bytes"], "min_document_bytes", 1)
    maximum = _integer(filters["max_document_bytes"], "max_document_bytes", minimum)
    normalized_filters = {
        **filters,
        **optional,
        **ratios,
        "min_document_bytes": minimum,
        "max_document_bytes": maximum,
        "maximum_spaced_OCR_sequences": _integer(
            filters["maximum_spaced_OCR_sequences"], "maximum_spaced_OCR_sequences"
        ),
        "minimum_science_term_hits": _integer(
            filters["minimum_science_term_hits"], "minimum_science_term_hits"
        ),
        "boilerplate_line_terms": _strings(
            filters["boilerplate_line_terms"], "boilerplate_line_terms"
        ),
        "accepted_document_licenses": _strings(
            filters["accepted_document_licenses"],
            "accepted_document_licenses",
            allow_empty=True,
        ),
        "science_terms": _strings(filters["science_terms"], "science_terms"),
        "maximum_bytes_per_host": (
            None
            if filters["maximum_bytes_per_host"] is None
            else _integer(filters["maximum_bytes_per_host"], "maximum_bytes_per_host", 1)
        ),
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
        partition["category"] != "science"
        or not isinstance(remainders, list)
        or not remainders
        or len(remainders) != len(set(remainders))
        or any(
            isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < modulus
            for value in remainders
        )
        or len(remainders) == modulus
    ):
        raise ValueError("invalid science downstream partition")
    normalized = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "status": config["status"],
        "source": normalized_source,
        "rights": rights,
        "filters": normalized_filters,
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


def load_science_sample_config(path):
    path = Path(path).resolve()
    return validate_science_sample_config(json.loads(path.read_text()), config_dir=path.parent)


def _field(row, reference):
    if reference is None:
        return None
    value = row
    for part in reference.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _compressed_rows(path, compression):
    if compression == "gzip":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    pending = b""
    with pa.input_stream(path, compression="zstd") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            lines = (pending + chunk).split(b"\n")
            pending = lines.pop()
            for line in lines:
                if line.strip():
                    yield json.loads(line)
    if pending.strip():
        yield json.loads(pending)


def _row_batches(source, seed):
    path = Path(source["path"])
    if source["file_format"] == "parquet":
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != source["rows"]:
            raise ValueError("science source row count mismatch")
        roots = {value.split(".", 1)[0] for value in source["fields"].values() if value}
        if roots - set(parquet.schema_arrow.names):
            raise ValueError("science source schema mismatch")
        order = sorted(
            range(parquet.num_row_groups),
            key=lambda index: hashlib.sha256(
                f"{seed}\0{source['id']}\0row-group\0{index}".encode()
            ).digest(),
        )
        for index in order:
            yield (
                f"row-group-{index}",
                parquet.read_row_group(index, columns=sorted(roots), use_threads=False).to_pylist(),
            )
        return
    compression = "gzip" if source["file_format"] == "jsonl_gzip" else "zstd"
    batch = []
    index = 0
    for row in _compressed_rows(path, compression):
        batch.append(row)
        if len(batch) == 4096:
            yield f"stream-batch-{index}", batch
            batch = []
            index += 1
    if batch:
        yield f"stream-batch-{index}", batch


def _boilerplate_ratio(text, terms):
    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    if not lines:
        return 1.0
    return sum(any(term in line for term in terms) for line in lines) / len(lines)


def _science_hits(text, terms):
    words = set(_WORD.findall(text.lower()))
    return sum(term in words for term in terms)


def _optional_string(value):
    return None if value is None else str(value)


def sample_science_source(config, *, restart=False):
    """Select a bounded science sample while preserving accepted document text."""

    if "plan_fingerprint" not in config:
        config = validate_science_sample_config(config)
    else:
        payload = {key: value for key, value in config.items() if key != "plan_fingerprint"}
        if config["plan_fingerprint"] != _fingerprint(payload):
            raise ValueError("normalized science sample fingerprint mismatch")
    source = config["source"]
    path = Path(source["path"])
    if not path.is_file() or _sha256(path) != source["sha256"]:
        raise ValueError("science source identity mismatch")
    output = Path(config["output_directory"])
    staging = output.with_name(output.name + ".building")
    if output.exists():
        raise FileExistsError(f"science sample already exists: {output}")
    if staging.exists():
        if not restart:
            raise FileExistsError(f"incomplete science sample exists: {staging}")
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    tokenizer_path = staging / "tokenizer-input.jsonl"
    attribution_path = staging / "attribution.jsonl"
    filters = config["filters"]
    settings = config["downstream_partition"]
    targets = {"train": settings["training_bytes"], "eval": settings["evaluation_bytes"]}
    counts = Counter()
    partitions = Counter()
    licenses = Counter()
    hosts = Counter()
    seen = set()
    batches_read = 0
    minimum_probability = None
    allowed_emails = {value.lower() for value in filters["allowed_email_placeholders"]}
    allowed_ips = set(filters["allowed_ipv4_placeholders"])
    accepted_licenses = set(filters["accepted_document_licenses"])
    with (
        tokenizer_path.open("w", encoding="utf-8") as tokenizer,
        attribution_path.open("w", encoding="utf-8") as attribution,
    ):
        for batch_id, rows in _row_batches(source, settings["seed"]):
            if all(partitions[f"{split}_bytes"] >= target for split, target in targets.items()):
                break
            batches_read += 1
            candidates = []
            for index, row in enumerate(rows):
                text = _field(row, source["fields"]["text"])
                if not isinstance(text, str):
                    counts["content_rejected"] += 1
                    continue
                digest = hashlib.sha256(text.encode()).hexdigest()
                priority = hashlib.sha256(
                    f"{settings['seed']}\0{source['id']}\0{digest}".encode()
                ).digest()
                candidates.append((priority, index, digest, row, text))
            for _, row_index, digest, row, text in sorted(candidates):
                if all(partitions[f"{split}_bytes"] >= target for split, target in targets.items()):
                    break
                counts["records_considered"] += 1
                fields = {name: _field(row, ref) for name, ref in source["fields"].items()}
                if (
                    filters["required_language"] is not None
                    and fields["language"] != filters["required_language"]
                ):
                    counts["metadata_language_rejected"] += 1
                    continue
                minimum_language = filters["minimum_language_score"]
                if minimum_language is not None and (
                    isinstance(fields["language_score"], bool)
                    or not isinstance(fields["language_score"], (int, float))
                    or fields["language_score"] < minimum_language
                ):
                    counts["metadata_language_score_rejected"] += 1
                    continue
                minimum_quality = filters["minimum_quality_score"]
                if minimum_quality is not None and (
                    isinstance(fields["quality_score"], bool)
                    or not isinstance(fields["quality_score"], (int, float))
                    or fields["quality_score"] < minimum_quality
                ):
                    counts["quality_score_rejected"] += 1
                    continue
                license_id = fields["license"]
                if accepted_licenses and license_id not in accepted_licenses:
                    counts["document_license_rejected"] += 1
                    continue
                raw = text.encode()
                size = len(raw)
                if not filters["min_document_bytes"] <= size <= filters["max_document_bytes"]:
                    counts["size_rejected"] += 1
                    continue
                if digest in seen:
                    counts["exact_duplicate_rejected"] += 1
                    continue
                if (
                    sum(character.isalpha() for character in text) / max(len(text), 1)
                    < filters["minimum_alphabetic_ratio"]
                ):
                    counts["alphabetic_ratio_rejected"] += 1
                    continue
                if _duplicate_line_ratio(text) > filters["maximum_duplicate_line_ratio"]:
                    counts["duplicate_lines_rejected"] += 1
                    continue
                if len(_SPACED_OCR.findall(text)) > filters["maximum_spaced_OCR_sequences"]:
                    counts["spaced_OCR_rejected"] += 1
                    continue
                if (
                    text.count("�") / max(len(text), 1)
                    > filters["maximum_replacement_character_ratio"]
                ):
                    counts["replacement_characters_rejected"] += 1
                    continue
                if (
                    _boilerplate_ratio(text, filters["boilerplate_line_terms"])
                    > filters["maximum_boilerplate_line_ratio"]
                ):
                    counts["boilerplate_rejected"] += 1
                    continue
                hits = _science_hits(text, filters["science_terms"])
                if hits < filters["minimum_science_term_hits"]:
                    counts["science_content_rejected"] += 1
                    continue
                emails, ips = _raw_pii(text, allowed_emails, allowed_ips)
                if emails or ips:
                    counts["raw_PII_rejected"] += 1
                    continue
                if sum(_secret_counts(text).values()):
                    counts["high_confidence_secret_rejected"] += 1
                    continue
                detected, probability = _language_identifier().classify(text)
                probability = float(probability)
                minimum_probability = (
                    probability
                    if minimum_probability is None
                    else min(minimum_probability, probability)
                )
                if (
                    detected != "en"
                    or probability < filters["minimum_detected_English_probability"]
                ):
                    counts["detected_non_English_rejected"] += 1
                    continue
                url = fields["url"] if isinstance(fields["url"], str) else None
                host = _host(url)
                host_limit = filters["maximum_bytes_per_host"]
                if host_limit is not None and host is not None and hosts[host] + size > host_limit:
                    counts["host_cap_rejected"] += 1
                    continue
                document_id = fields["document_id"]
                if document_id is None:
                    document_id = hashlib.sha256(
                        f"{source['sha256']}\0{batch_id}\0{row_index}".encode()
                    ).hexdigest()
                record = {
                    "text": text,
                    "source": source["id"],
                    "content_id": str(document_id),
                    "released_content_sha256": digest,
                    "repo_path": host or source["repo"],
                    "repo_id": None,
                    "commit_id": source["revision"],
                    "file_path": url or f"{path.name}:{batch_id}:{row_index}",
                    "language": "English",
                    "detected_licenses": [str(license_id)] if license_id is not None else [],
                    "rights_status": "paper_or_document_rights_manual_review_required",
                    "url": url,
                    "host": host,
                    "title": _optional_string(fields["title"]),
                    "publication_date": _optional_string(fields["date"]),
                    "source_partition": _optional_string(fields["source_partition"]),
                    "size_bytes": size,
                    "detected_English_probability": probability,
                    "science_term_hits": hits,
                }
                partition = _sample_partition(record, settings)
                if partitions[f"{partition}_bytes"] >= targets[partition]:
                    counts[f"{partition}_quota_already_filled"] += 1
                    continue
                _dump_line(tokenizer, record)
                _dump_line(
                    attribution, {key: value for key, value in record.items() if key != "text"}
                )
                seen.add(digest)
                if host is not None:
                    hosts[host] += size
                if license_id is not None:
                    licenses[str(license_id)] += 1
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
        "source": source,
        "rights": config["rights"],
        "filters": filters,
        "counts": dict(sorted(counts.items())),
        "source_profile": {
            "declared_rows": source["rows"],
            "batches_read": batches_read,
            "selected_hosts": len(hosts),
            "largest_selected_host_bytes": max(hosts.values(), default=0),
            "minimum_observed_detected_English_probability": minimum_probability,
            "content_preservation": "accepted source text copied byte-for-byte",
        },
        "accepted_document_licenses": dict(licenses.most_common()),
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
            "source_identity_schema_and_declared_rows": "pass",
            "paper_or_document_identity": "pass",
            "document_license_metadata_policy": "pass",
            "English_and_science_content": "pass",
            "OCR_boilerplate_repetition_and_encoding": "pass",
            "conservative_PII_and_secret_filter": "pass",
            "downstream_tokenizer_partition_yield": "fail" if missing else "pass",
            "dedicated_secret_scanner": "pending",
            "cross_source_near_duplicates": "pending",
            "benchmark_contamination": "pending_evaluation_firewall",
            "manual_legal_acceptance": "pending",
            "training_authority": "blocked",
        },
    }
    if missing:
        report["missing_downstream_partition_bytes"] = missing
    _write_json(staging / "report.json", report)
    if missing:
        raise RuntimeError(f"science source cannot fill tokenizer partitions: {missing}")
    os.replace(staging, output)
    return report
