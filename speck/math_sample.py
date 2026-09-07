"""Build deterministic bounded math samples without rewriting mathematical notation."""

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
from speck.stack_v3_refine import _english_prose_result, _language_identifier, _sample_partition
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

FORMAT = "speck_math_sample"
FORMAT_VERSION = 1
REPORT_FORMAT = "speck_math_sample_result"
_URL = re.compile(r"https?://\S+")
_FENCED = re.compile(r"```.*?```", re.DOTALL)
_DISPLAY_MATH = re.compile(r"\$\$.*?\$\$|\\\[.*?\\\]", re.DOTALL)
_INLINE_MATH = re.compile(r"(?<!\$)\$(?!\$).*?(?<!\$)\$(?!\$)", re.DOTALL)
_LATEX_COMMAND = re.compile(r"\\[A-Za-z]+\*?")
_LATEX_SIGNAL = re.compile(r"\$|\\(?:begin|end|frac|sum|int|sqrt|alpha|beta|gamma|theta|pi)\b")


def validate_math_sample_config(config, *, config_dir=None):
    """Validate and normalize one immutable bounded math-source plan."""

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
        "math sample",
    )
    if config["format"] != FORMAT or config["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported math sample format")
    if config["status"] != "bounded_sample_authorized_not_training_authority":
        raise ValueError("math sample must remain non-authoritative")

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
            "text_field",
            "metadata_json_field",
            "fields",
        },
        "source",
    )
    for key in ("id", "repo", "revision", "official_url", "text_field"):
        if not isinstance(source[key], str) or not source[key]:
            raise ValueError(f"source.{key} must be non-empty")
    if Path(source["id"]).name != source["id"]:
        raise ValueError("source.id must be a path component")
    if source["file_format"] not in {"parquet", "jsonl_zstd"}:
        raise ValueError("source.file_format is unsupported")
    metadata_field = source["metadata_json_field"]
    if metadata_field is not None and (not isinstance(metadata_field, str) or not metadata_field):
        raise ValueError("source.metadata_json_field must be null or a string")
    fields = source["fields"]
    _exact_keys(
        fields,
        {
            "document_id",
            "url",
            "language",
            "language_score",
            "quality_score",
            "code_language",
            "detected_licenses",
            "license_type",
            "repository",
            "file_path",
        },
        "source.fields",
    )
    for key, value in fields.items():
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"source.fields.{key} must be null or a string")
        if isinstance(value, str) and value.startswith("$metadata.") and metadata_field is None:
            raise ValueError(f"source.fields.{key} requires metadata_json_field")
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
        raise ValueError("math rights must remain a non-empty manual-review gate")

    filters = config["filters"]
    _exact_keys(
        filters,
        {
            "language_policy",
            "required_language",
            "minimum_language_score",
            "minimum_quality_score",
            "minimum_prose_alphabetic_characters",
            "minimum_detected_English_probability",
            "min_document_bytes",
            "max_document_bytes",
            "maximum_duplicate_line_ratio",
            "maximum_bytes_per_host",
            "allowed_email_placeholders",
            "allowed_ipv4_placeholders",
            "language_detector",
            "accepted_detected_licenses",
            "required_license_type",
        },
        "filters",
    )
    if filters["language_policy"] not in {
        "metadata_and_math_prose",
        "math_prose",
        "code_comments_and_docstrings",
    }:
        raise ValueError("unsupported math language policy")
    if filters["language_detector"] != "py3langid==0.3.0":
        raise ValueError("math language detector must be pinned py3langid")
    if filters["required_language"] is not None and (
        not isinstance(filters["required_language"], str) or not filters["required_language"]
    ):
        raise ValueError("required_language must be null or a string")
    numeric_optional = {}
    for key in (
        "minimum_language_score",
        "minimum_quality_score",
        "maximum_bytes_per_host",
    ):
        value = filters[key]
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0
        ):
            raise ValueError(f"filters.{key} must be null or non-negative")
        numeric_optional[key] = None if value is None else float(value)
    for key in ("minimum_detected_English_probability", "maximum_duplicate_line_ratio"):
        value = filters[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"filters.{key} must be in [0, 1]")
    minimum_size = _integer(filters["min_document_bytes"], "min_document_bytes", 1)
    maximum_size = _integer(filters["max_document_bytes"], "max_document_bytes", minimum_size)
    normalized_filters = {
        **filters,
        **numeric_optional,
        "minimum_prose_alphabetic_characters": _integer(
            filters["minimum_prose_alphabetic_characters"],
            "minimum_prose_alphabetic_characters",
            1,
        ),
        "minimum_detected_English_probability": float(
            filters["minimum_detected_English_probability"]
        ),
        "min_document_bytes": minimum_size,
        "max_document_bytes": maximum_size,
        "maximum_duplicate_line_ratio": float(filters["maximum_duplicate_line_ratio"]),
        "allowed_email_placeholders": _strings(
            filters["allowed_email_placeholders"], "allowed_email_placeholders", allow_empty=True
        ),
        "allowed_ipv4_placeholders": _strings(
            filters["allowed_ipv4_placeholders"], "allowed_ipv4_placeholders", allow_empty=True
        ),
        "accepted_detected_licenses": _strings(
            filters["accepted_detected_licenses"],
            "accepted_detected_licenses",
            allow_empty=True,
        ),
    }
    if filters["required_license_type"] is not None and (
        not isinstance(filters["required_license_type"], str)
        or not filters["required_license_type"]
    ):
        raise ValueError("required_license_type must be null or a string")

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
        raise ValueError("invalid math downstream partition")
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


def load_math_sample_config(path):
    path = Path(path).resolve()
    return validate_math_sample_config(json.loads(path.read_text()), config_dir=path.parent)


def _field(row, reference, metadata):
    if reference is None:
        return None
    if reference.startswith("$metadata."):
        return metadata.get(reference.removeprefix("$metadata."))
    return row.get(reference)


def _math_prose(text):
    text = _FENCED.sub(" ", text)
    text = _DISPLAY_MATH.sub(" ", text)
    text = _INLINE_MATH.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _LATEX_COMMAND.sub(" ", text)
    return " ".join(text.split())


def _language_result(text, code_language, filters):
    if filters["language_policy"] == "code_comments_and_docstrings":
        return _english_prose_result(
            text,
            code_language or "C++",
            {
                "minimum_alphabetic_characters": filters["minimum_prose_alphabetic_characters"],
                "minimum_probability": filters["minimum_detected_English_probability"],
            },
        )
    prose = _math_prose(text)
    if (
        sum(character.isalpha() for character in prose)
        < filters["minimum_prose_alphabetic_characters"]
    ):
        return "insufficient_prose", None
    detected, probability = _language_identifier().classify(prose)
    probability = float(probability)
    if detected == "en" and probability >= filters["minimum_detected_English_probability"]:
        return "English", probability
    return "non_English", probability


def _jsonl_zstd_batches(path, batch_size=4096):
    batches = []
    batch = []
    pending = b""
    with pa.input_stream(path, compression="zstd") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            lines = (pending + chunk).split(b"\n")
            pending = lines.pop()
            for raw in lines:
                if not raw.strip():
                    continue
                batch.append(json.loads(raw.decode("utf-8")))
                if len(batch) == batch_size:
                    batches.append(batch)
                    batch = []
    if pending.strip():
        batch.append(json.loads(pending.decode("utf-8")))
    if batch:
        batches.append(batch)
    return batches


def _source_batches(source, settings):
    path = Path(source["path"])
    if source["file_format"] == "parquet":
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != source["rows"]:
            raise ValueError("math source row count mismatch")
        required = {source["text_field"]}
        if source["metadata_json_field"]:
            required.add(source["metadata_json_field"])
        required.update(
            reference
            for reference in source["fields"].values()
            if isinstance(reference, str) and not reference.startswith("$metadata.")
        )
        if required - set(parquet.schema_arrow.names):
            raise ValueError("math source schema mismatch")
        order = sorted(
            range(parquet.num_row_groups),
            key=lambda index: hashlib.sha256(
                f"{settings['seed']}\0{source['id']}\0row-group\0{index}".encode()
            ).digest(),
        )
        return (
            (
                f"row-group-{index}",
                parquet.read_row_group(
                    index, columns=sorted(required), use_threads=False
                ).to_pylist(),
            )
            for index in order
        ), {"rows_total": parquet.metadata.num_rows, "batches_total": parquet.num_row_groups}

    batches = _jsonl_zstd_batches(path)
    rows = sum(len(batch) for batch in batches)
    if rows != source["rows"]:
        raise ValueError("math source row count mismatch")
    required = {source["text_field"]}
    if source["metadata_json_field"]:
        required.add(source["metadata_json_field"])
    required.update(
        reference
        for reference in source["fields"].values()
        if isinstance(reference, str) and not reference.startswith("$metadata.")
    )
    if any(required - set(row) for batch in batches for row in batch):
        raise ValueError("math source schema mismatch")
    order = sorted(
        range(len(batches)),
        key=lambda index: hashlib.sha256(
            f"{settings['seed']}\0{source['id']}\0batch\0{index}".encode()
        ).digest(),
    )
    return ((f"batch-{index}", batches[index]) for index in order), {
        "rows_total": rows,
        "batches_total": len(batches),
    }


def sample_math_source(config, *, restart=False):
    """Select a bounded source while preserving accepted text byte-for-byte."""

    if "plan_fingerprint" not in config:
        config = validate_math_sample_config(config)
    else:
        payload = {key: value for key, value in config.items() if key != "plan_fingerprint"}
        if config["plan_fingerprint"] != _fingerprint(payload):
            raise ValueError("normalized math sample fingerprint mismatch")
    source = config["source"]
    source_path = Path(source["path"])
    if not source_path.is_file() or _sha256(source_path) != source["sha256"]:
        raise ValueError("math source identity mismatch")
    settings = config["downstream_partition"]
    batches, profile = _source_batches(source, settings)

    output = Path(config["output_directory"])
    staging = output.with_name(output.name + ".building")
    if output.exists():
        raise FileExistsError(f"math sample already exists: {output}")
    if staging.exists():
        if not restart:
            raise FileExistsError(f"incomplete math sample exists: {staging}")
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    tokenizer_path = staging / "tokenizer-input.jsonl"
    attribution_path = staging / "attribution.jsonl"
    counts = Counter()
    partitions = Counter()
    host_bytes = Counter()
    probabilities = []
    seen = set()
    targets = {"train": settings["training_bytes"], "eval": settings["evaluation_bytes"]}
    filters = config["filters"]
    allowed_emails = {value.lower() for value in filters["allowed_email_placeholders"]}
    allowed_ips = set(filters["allowed_ipv4_placeholders"])
    batches_read = 0
    with (
        tokenizer_path.open("w", encoding="utf-8") as tokenizer,
        attribution_path.open("w", encoding="utf-8") as attribution,
    ):
        for batch_id, rows in batches:
            if all(partitions[f"{split}_bytes"] >= target for split, target in targets.items()):
                break
            batches_read += 1
            candidates = []
            for row_index, row in enumerate(rows):
                text = row.get(source["text_field"])
                if not isinstance(text, str):
                    counts["content_rejected"] += 1
                    continue
                digest = hashlib.sha256(text.encode()).hexdigest()
                priority = hashlib.sha256(
                    f"{settings['seed']}\0{source['id']}\0{digest}".encode()
                ).digest()
                candidates.append((priority, row_index, digest, row, text))
            for _, row_index, digest, row, text in sorted(candidates):
                if all(partitions[f"{split}_bytes"] >= target for split, target in targets.items()):
                    break
                counts["records_considered"] += 1
                metadata = {}
                if source["metadata_json_field"]:
                    raw_metadata = row[source["metadata_json_field"]]
                    try:
                        metadata = (
                            raw_metadata
                            if isinstance(raw_metadata, dict)
                            else json.loads(raw_metadata)
                        )
                    except (TypeError, json.JSONDecodeError):
                        counts["metadata_json_rejected"] += 1
                        continue
                fields = {
                    name: _field(row, reference, metadata)
                    for name, reference in source["fields"].items()
                }
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
                detected_licenses = fields["detected_licenses"] or []
                accepted_licenses = filters["accepted_detected_licenses"]
                if not isinstance(detected_licenses, list) or any(
                    not isinstance(value, str) or not value for value in detected_licenses
                ):
                    counts["license_metadata_rejected"] += 1
                    continue
                if accepted_licenses and (
                    not detected_licenses
                    or any(value not in accepted_licenses for value in detected_licenses)
                ):
                    counts["license_allowlist_rejected"] += 1
                    continue
                required_license_type = filters["required_license_type"]
                if (
                    required_license_type is not None
                    and fields["license_type"] != required_license_type
                ):
                    counts["license_type_rejected"] += 1
                    continue
                raw = text.encode()
                size = len(raw)
                if not filters["min_document_bytes"] <= size <= filters["max_document_bytes"]:
                    counts["size_rejected"] += 1
                    continue
                if _duplicate_line_ratio(text) > filters["maximum_duplicate_line_ratio"]:
                    counts["duplicate_lines_rejected"] += 1
                    continue
                url = fields["url"] if isinstance(fields["url"], str) else None
                host = _host(url)
                host_limit = filters["maximum_bytes_per_host"]
                if url is not None and host is None:
                    counts["url_rejected"] += 1
                    continue
                if (
                    host_limit is not None
                    and host is not None
                    and host_bytes[host] + size > host_limit
                ):
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
                language_result, probability = _language_result(
                    text, fields["code_language"], filters
                )
                if probability is not None:
                    probabilities.append(probability)
                if language_result == "non_English":
                    counts["detected_non_English_prose_rejected"] += 1
                    continue
                if language_result == "insufficient_prose":
                    counts["notation_or_code_without_enough_prose"] += 1
                content_id = fields["document_id"]
                if content_id is None:
                    content_id = hashlib.sha256(
                        f"{source['sha256']}\0{batch_id}\0{row_index}".encode()
                    ).hexdigest()
                record = {
                    "text": text,
                    "source": source["id"],
                    "content_id": str(content_id),
                    "released_content_sha256": digest,
                    "repo_path": fields["repository"] or host or source["repo"],
                    "repo_id": None,
                    "commit_id": source["revision"],
                    "file_path": fields["file_path"]
                    or url
                    or f"{source_path.name}:{batch_id}:{row_index}",
                    "language": fields["code_language"] or "English",
                    "detected_licenses": detected_licenses,
                    "rights_status": "math_source_terms_manual_review_required",
                    "url": url,
                    "host": host,
                    "size_bytes": size,
                    "language_score": fields["language_score"],
                    "detected_English_probability": probability,
                    "quality_score": fields["quality_score"],
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
                    host_bytes[host] += size
                partitions[f"{partition}_bytes"] += size
                partitions[f"{partition}_records"] += 1
                counts["records_sampled"] += 1
                if _LATEX_SIGNAL.search(text):
                    counts["records_with_latex_signal"] += 1
                    counts["bytes_with_latex_signal"] += size
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
            **profile,
            "batches_read": batches_read,
            "selected_hosts": len(host_bytes),
            "largest_selected_host_bytes": max(host_bytes.values(), default=0),
            "minimum_observed_detected_English_probability": min(probabilities, default=None),
            "content_preservation": "accepted source text copied byte-for-byte",
        },
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
            "source_identity_schema_and_rows": "pass",
            "source_native_quality": "pass",
            "English_prose_with_math_and_code_exemption": "pass",
            "LaTeX_and_code_notation_preserved": "pass",
            "conservative_PII_secret_host_and_repetition_filters": "pass",
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
        raise RuntimeError(f"math source cannot fill tokenizer partitions: {missing}")
    os.replace(staging, output)
    return report
