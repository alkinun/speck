"""Build deterministic synthetic-text samples with generator and seed lineage."""

import hashlib
import json
import math
import os
import re
import shutil
from collections import Counter
from pathlib import Path

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

FORMAT = "speck_synthetic_sample"
FORMAT_VERSION = 1
REPORT_FORMAT = "speck_synthetic_sample_result"
_TOKENS = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


def _nonempty(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be non-empty")
    return value


def validate_synthetic_sample_config(config, *, config_dir=None):
    """Validate one frozen multi-input synthetic source plan."""

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
        "synthetic sample",
    )
    if config["format"] != FORMAT or config["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported synthetic sample format")
    if config["status"] != "bounded_sample_authorized_not_training_authority":
        raise ValueError("synthetic sample must remain non-authoritative")
    source = config["source"]
    _exact_keys(
        source,
        {"id", "repo", "revision", "official_url", "inputs"},
        "source",
    )
    for key in ("id", "repo", "revision", "official_url"):
        _nonempty(source[key], f"source.{key}")
    if Path(source["id"]).name != source["id"]:
        raise ValueError("source.id must be a path component")
    inputs = source["inputs"]
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("source.inputs must be non-empty")
    normalized_inputs = []
    input_ids = []
    for index, item in enumerate(inputs):
        _exact_keys(
            item,
            {
                "id",
                "path",
                "sha256",
                "rows",
                "format",
                "training_bytes",
                "evaluation_bytes",
                "generator",
                "transformation",
                "seed_source",
                "fields",
                "maximum_seed_text_jaccard",
            },
            f"input {index}",
        )
        input_id = _nonempty(item["id"], f"input {index} id")
        if Path(input_id).name != input_id:
            raise ValueError("input ID must be a path component")
        input_ids.append(input_id)
        if item["format"] != "parquet":
            raise ValueError("synthetic source inputs must be Parquet")
        generator = item["generator"]
        _exact_keys(
            generator,
            {"id", "revision", "official_url", "license"},
            f"input {input_id} generator",
        )
        seed = item["seed_source"]
        _exact_keys(
            seed,
            {"id", "revision", "official_url", "rights"},
            f"input {input_id} seed_source",
        )
        for group_name, group in (("generator", generator), ("seed_source", seed)):
            for key, value in group.items():
                _nonempty(value, f"input {input_id} {group_name}.{key}")
        _nonempty(item["transformation"], f"input {input_id} transformation")
        fields = item["fields"]
        old_fields = {
            "text",
            "document_id",
            "prompt",
            "seed",
            "style",
            "answer",
            "url",
            "domain",
        }
        if frozenset(fields) not in {
            frozenset(old_fields),
            frozenset(old_fields | {"seed_label"}),
        }:
            raise ValueError(f"input {input_id} fields have an unsupported schema")
        _nonempty(fields["text"], f"input {input_id} fields.text")
        for key, value in fields.items():
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"input {input_id} field {key} must be null or a string")
        normalized_inputs.append(
            {
                **item,
                "path": _path(item["path"], f"input {input_id} path", config_dir),
                "sha256": _digest(item["sha256"], f"input {input_id} sha256"),
                "rows": _integer(item["rows"], f"input {input_id} rows", 1),
                "training_bytes": _integer(
                    item["training_bytes"], f"input {input_id} training_bytes", 1
                ),
                "evaluation_bytes": _integer(
                    item["evaluation_bytes"], f"input {input_id} evaluation_bytes", 1
                ),
                "maximum_seed_text_jaccard": (
                    None
                    if item["maximum_seed_text_jaccard"] is None
                    else float(item["maximum_seed_text_jaccard"])
                ),
            }
        )
        seed_overlap = item["maximum_seed_text_jaccard"]
        if seed_overlap is not None and (
            isinstance(seed_overlap, bool)
            or not isinstance(seed_overlap, (int, float))
            or not 0 <= seed_overlap <= 1
        ):
            raise ValueError(
                f"input {input_id} maximum_seed_text_jaccard must be null or in [0, 1]"
            )
    if len(input_ids) != len(set(input_ids)):
        raise ValueError("synthetic input IDs must be unique")

    rights = config["rights"]
    _exact_keys(
        rights,
        {"dataset_license", "upstream_terms", "authority", "redistribution"},
        "rights",
    )
    if rights["authority"] != "manual_review_required" or any(
        not isinstance(value, str) or not value for value in rights.values()
    ):
        raise ValueError("synthetic rights must remain a manual-review gate")
    filters = config["filters"]
    _exact_keys(
        filters,
        {
            "language_detector",
            "minimum_detected_English_probability",
            "min_document_bytes",
            "max_document_bytes",
            "maximum_duplicate_line_ratio",
            "repetition_ngram_tokens",
            "maximum_repeated_ngram_ratio",
            "template_prefix_tokens",
            "maximum_bytes_per_template_prefix",
            "model_identity_phrases",
            "allowed_email_placeholders",
            "allowed_ipv4_placeholders",
            "seed_overlap_shingle_tokens",
            "maximum_bytes_per_seed_domain",
        },
        "filters",
    )
    if filters["language_detector"] != "py3langid==0.3.0":
        raise ValueError("synthetic language detector must be pinned py3langid")
    ratios = {}
    for key in (
        "minimum_detected_English_probability",
        "maximum_duplicate_line_ratio",
        "maximum_repeated_ngram_ratio",
    ):
        value = filters[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"filters.{key} must be in [0, 1]")
        ratios[key] = float(value)
    minimum = _integer(filters["min_document_bytes"], "min_document_bytes", 1)
    maximum = _integer(filters["max_document_bytes"], "max_document_bytes", minimum)
    normalized_filters = {
        **filters,
        **ratios,
        "min_document_bytes": minimum,
        "max_document_bytes": maximum,
        "repetition_ngram_tokens": _integer(
            filters["repetition_ngram_tokens"], "repetition_ngram_tokens", 2
        ),
        "template_prefix_tokens": _integer(
            filters["template_prefix_tokens"], "template_prefix_tokens", 2
        ),
        "seed_overlap_shingle_tokens": _integer(
            filters["seed_overlap_shingle_tokens"], "seed_overlap_shingle_tokens", 2
        ),
        "maximum_bytes_per_seed_domain": (
            None
            if filters["maximum_bytes_per_seed_domain"] is None
            else _integer(
                filters["maximum_bytes_per_seed_domain"],
                "maximum_bytes_per_seed_domain",
                1,
            )
        ),
        "maximum_bytes_per_template_prefix": _integer(
            filters["maximum_bytes_per_template_prefix"],
            "maximum_bytes_per_template_prefix",
            1,
        ),
        "model_identity_phrases": _strings(
            filters["model_identity_phrases"], "model_identity_phrases"
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
        {"seed", "category", "modulus", "evaluation_remainders"},
        "downstream_partition",
    )
    modulus = _integer(partition["modulus"], "partition.modulus", 2)
    remainders = partition["evaluation_remainders"]
    if (
        partition["category"] != "synthetic"
        or not isinstance(remainders, list)
        or not remainders
        or len(remainders) != len(set(remainders))
        or any(
            isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < modulus
            for value in remainders
        )
        or len(remainders) == modulus
    ):
        raise ValueError("invalid synthetic downstream partition")
    normalized = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "status": config["status"],
        "source": {**source, "inputs": normalized_inputs},
        "rights": rights,
        "filters": normalized_filters,
        "downstream_partition": {
            "seed": _integer(partition["seed"], "partition.seed"),
            "category": "synthetic",
            "modulus": modulus,
            "evaluation_remainders": sorted(remainders),
        },
        "output_directory": _path(config["output_directory"], "output_directory", config_dir),
    }
    normalized["plan_fingerprint"] = _fingerprint(normalized)
    return normalized


def load_synthetic_sample_config(path):
    path = Path(path).resolve()
    return validate_synthetic_sample_config(json.loads(path.read_text()), config_dir=path.parent)


def _tokens(text):
    return _TOKENS.findall(text.lower())


def _repeated_ngram_ratio(text, size):
    tokens = _tokens(text)
    total = len(tokens) - size + 1
    if total <= 0:
        return 0.0
    ngrams = {tuple(tokens[index : index + size]) for index in range(total)}
    return 1 - len(ngrams) / total


def _template_prefix(text, size):
    return " ".join(_tokens(text)[:size])


def _shingle_jaccard(left, right, size):
    def shingles(value):
        tokens = _tokens(value)
        return {tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}

    left_shingles = shingles(left)
    right_shingles = shingles(right)
    union = left_shingles | right_shingles
    return len(left_shingles & right_shingles) / len(union) if union else 0.0


def _entropy(counter):
    total = sum(counter.values())
    if not total:
        return None
    return -sum((count / total) * math.log2(count / total) for count in counter.values())


def sample_synthetic_source(config, *, restart=False):
    """Select source-balanced synthetic text and preserve lineage without copying seeds."""

    if "plan_fingerprint" not in config:
        config = validate_synthetic_sample_config(config)
    else:
        payload = {key: value for key, value in config.items() if key != "plan_fingerprint"}
        if config["plan_fingerprint"] != _fingerprint(payload):
            raise ValueError("normalized synthetic sample fingerprint mismatch")
    for item in config["source"]["inputs"]:
        path = Path(item["path"])
        parquet = pq.ParquetFile(path)
        required = {value for value in item["fields"].values() if value is not None}
        if (
            not path.is_file()
            or _sha256(path) != item["sha256"]
            or parquet.metadata.num_rows != item["rows"]
            or required - set(parquet.schema_arrow.names)
        ):
            raise ValueError(f"synthetic input identity/schema mismatch: {item['id']}")

    output = Path(config["output_directory"])
    staging = output.with_name(output.name + ".building")
    if output.exists():
        raise FileExistsError(f"synthetic sample already exists: {output}")
    if staging.exists():
        if not restart:
            raise FileExistsError(f"incomplete synthetic sample exists: {staging}")
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    tokenizer_path = staging / "tokenizer-input.jsonl"
    attribution_path = staging / "attribution.jsonl"
    filters = config["filters"]
    settings = config["downstream_partition"]
    seen = set()
    template_bytes = Counter()
    domain_bytes = Counter()
    aggregate = Counter()
    styles = Counter()
    input_reports = []
    allowed_emails = {value.lower() for value in filters["allowed_email_placeholders"]}
    allowed_ips = set(filters["allowed_ipv4_placeholders"])
    phrases = [value.lower() for value in filters["model_identity_phrases"]]
    with (
        tokenizer_path.open("w", encoding="utf-8") as tokenizer,
        attribution_path.open("w", encoding="utf-8") as attribution,
    ):
        for item in config["source"]["inputs"]:
            parquet = pq.ParquetFile(item["path"])
            columns = sorted({value for value in item["fields"].values() if value is not None})
            targets = {"train": item["training_bytes"], "eval": item["evaluation_bytes"]}
            counts = Counter()
            partitions = Counter()
            maximum_seed_overlap = 0.0
            row_groups = sorted(
                range(parquet.num_row_groups),
                key=lambda index: hashlib.sha256(
                    f"{settings['seed']}\0{config['source']['id']}\0{item['id']}\0{index}".encode()
                ).digest(),
            )
            groups_read = 0
            for row_group in row_groups:
                if all(partitions[f"{split}_bytes"] >= target for split, target in targets.items()):
                    break
                groups_read += 1
                rows = parquet.read_row_group(
                    row_group, columns=columns, use_threads=False
                ).to_pylist()
                candidates = []
                for row_index, row in enumerate(rows):
                    text = row.get(item["fields"]["text"])
                    if not isinstance(text, str):
                        counts["content_rejected"] += 1
                        continue
                    digest = hashlib.sha256(text.encode()).hexdigest()
                    priority = hashlib.sha256(
                        f"{settings['seed']}\0{config['source']['id']}\0{item['id']}\0{digest}".encode()
                    ).digest()
                    candidates.append((priority, row_index, digest, row, text))
                for _, row_index, digest, row, text in sorted(candidates):
                    if all(
                        partitions[f"{split}_bytes"] >= target for split, target in targets.items()
                    ):
                        break
                    counts["records_considered"] += 1
                    size = len(text.encode())
                    if not filters["min_document_bytes"] <= size <= filters["max_document_bytes"]:
                        counts["size_rejected"] += 1
                        continue
                    if digest in seen:
                        counts["exact_duplicate_rejected"] += 1
                        continue
                    if _duplicate_line_ratio(text) > filters["maximum_duplicate_line_ratio"]:
                        counts["duplicate_lines_rejected"] += 1
                        continue
                    repetition = _repeated_ngram_ratio(text, filters["repetition_ngram_tokens"])
                    if repetition > filters["maximum_repeated_ngram_ratio"]:
                        counts["repeated_ngrams_rejected"] += 1
                        continue
                    lower = text.lower()
                    if any(phrase in lower for phrase in phrases):
                        counts["model_identity_phrase_rejected"] += 1
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
                    if (
                        detected != "en"
                        or probability < filters["minimum_detected_English_probability"]
                    ):
                        counts["detected_non_English_rejected"] += 1
                        continue
                    template = _template_prefix(text, filters["template_prefix_tokens"])
                    if not template:
                        counts["empty_template_rejected"] += 1
                        continue
                    if (
                        template_bytes[template] + size
                        > filters["maximum_bytes_per_template_prefix"]
                    ):
                        counts["template_prefix_cap_rejected"] += 1
                        continue
                    fields = item["fields"]
                    prompt = row.get(fields["prompt"]) if fields["prompt"] else None
                    seed = row.get(fields["seed"]) if fields["seed"] else None
                    seed_label_field = fields.get("seed_label")
                    seed_label = row.get(seed_label_field) if seed_label_field else None
                    answer = row.get(fields["answer"]) if fields["answer"] else None
                    url = row.get(fields["url"]) if fields["url"] else None
                    domain = row.get(fields["domain"]) if fields["domain"] else _host(url)
                    if domain is not None and (not isinstance(domain, str) or not domain):
                        counts["seed_domain_rejected"] += 1
                        continue
                    domain_limit = filters["maximum_bytes_per_seed_domain"]
                    if (
                        domain_limit is not None
                        and domain is not None
                        and domain_bytes[domain] + size > domain_limit
                    ):
                        counts["seed_domain_cap_rejected"] += 1
                        continue
                    style = row.get(fields["style"]) if fields["style"] else item["transformation"]
                    document_id = row.get(fields["document_id"]) if fields["document_id"] else None
                    if document_id is None:
                        document_id = hashlib.sha256(
                            f"{item['sha256']}\0{row_group}\0{row_index}".encode()
                        ).hexdigest()
                    seed_overlap = (
                        _shingle_jaccard(text, seed, filters["seed_overlap_shingle_tokens"])
                        if isinstance(seed, str)
                        else None
                    )
                    if seed_overlap is not None:
                        maximum_seed_overlap = max(maximum_seed_overlap, seed_overlap)
                    if (
                        item["maximum_seed_text_jaccard"] is not None
                        and seed_overlap is not None
                        and seed_overlap >= item["maximum_seed_text_jaccard"]
                    ):
                        counts["unchanged_seed_rejected"] += 1
                        continue
                    record = {
                        "text": text,
                        "source": config["source"]["id"],
                        "source_input": item["id"],
                        "content_id": str(document_id),
                        "released_content_sha256": digest,
                        "repo_path": config["source"]["repo"],
                        "repo_id": None,
                        "commit_id": config["source"]["revision"],
                        "file_path": f"{Path(item['path']).name}:{row_group}:{row_index}",
                        "language": "English",
                        "detected_licenses": [],
                        "rights_status": "synthetic_generator_and_seed_terms_manual_review_required",
                        "url": url if isinstance(url, str) else None,
                        "host": domain,
                        "size_bytes": size,
                        "generator_id": item["generator"]["id"],
                        "generator_revision": item["generator"]["revision"],
                        "generator_license": item["generator"]["license"],
                        "transformation": item["transformation"],
                        "seed_source_id": item["seed_source"]["id"],
                        "seed_source_revision": item["seed_source"]["revision"],
                        "seed_source_label": (str(seed_label) if seed_label is not None else None),
                        "prompt_sha256": (
                            hashlib.sha256(prompt.encode()).hexdigest()
                            if isinstance(prompt, str)
                            else None
                        ),
                        "seed_sha256": (
                            hashlib.sha256(seed.encode()).hexdigest()
                            if isinstance(seed, str)
                            else None
                        ),
                        "seed_text_shingle_jaccard": seed_overlap,
                        "answer_sha256": (
                            hashlib.sha256(answer.encode()).hexdigest()
                            if isinstance(answer, str)
                            else None
                        ),
                        "style": str(style) if style is not None else item["transformation"],
                        "detected_English_probability": probability,
                        "repeated_ngram_ratio": repetition,
                        "template_prefix_sha256": hashlib.sha256(template.encode()).hexdigest(),
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
                    template_bytes[template] += size
                    if domain is not None:
                        domain_bytes[domain] += size
                    styles[record["style"]] += 1
                    partitions[f"{partition}_bytes"] += size
                    partitions[f"{partition}_records"] += 1
                    counts["records_sampled"] += 1
                    counts["records_with_prompt_lineage"] += isinstance(prompt, str)
                    counts["records_with_seed_lineage"] += isinstance(seed, str)
                    counts["records_with_answer_lineage"] += isinstance(answer, str)
            aggregate.update(counts)
            input_reports.append(
                {
                    "id": item["id"],
                    "input": item,
                    "counts": dict(sorted(counts.items())),
                    "row_groups_total": parquet.num_row_groups,
                    "row_groups_read": groups_read,
                    "maximum_observed_seed_text_jaccard": maximum_seed_overlap,
                    "partition": {**dict(sorted(partitions.items())), "targets": targets},
                }
            )
        for handle in (tokenizer, attribution):
            handle.flush()
            os.fsync(handle.fileno())
    missing = [
        report["id"]
        for report in input_reports
        if any(
            report["partition"].get(f"{split}_bytes", 0) < target
            for split, target in report["partition"]["targets"].items()
        )
    ]
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
        "filters": filters,
        "counts": dict(sorted(aggregate.items())),
        "inputs": input_reports,
        "style_profile": {
            "counts": dict(styles.most_common()),
            "Shannon_entropy_bits": _entropy(styles),
            "unique_styles": len(styles),
            "unique_template_prefixes": len(template_bytes),
            "largest_template_prefix_bytes": max(template_bytes.values(), default=0),
            "unique_seed_domains": len(domain_bytes),
            "largest_seed_domain_bytes": max(domain_bytes.values(), default=0),
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
            "generator_and_seed_lineage_recorded": "pass_with_undisclosed_revisions_preserved",
            "English_language": "pass",
            "within_document_repetition": "pass",
            "cross_document_template_concentration": "pass",
            "model_identity_and_refusal_phrases": "pass",
            "conservative_PII_and_secret_filter": "pass",
            "per_input_partition_yield": "fail" if missing else "pass",
            "dedicated_secret_scanner": "pending",
            "cross_source_near_duplicates": "pending",
            "benchmark_and_answer_contamination": "pending_evaluation_firewall",
            "manual_generator_seed_and_dataset_rights": "pending",
            "training_authority": "blocked",
        },
    }
    if missing:
        report["inputs_missing_partition_quota"] = missing
    _write_json(staging / "report.json", report)
    if missing:
        raise RuntimeError(f"synthetic inputs cannot fill partition quotas: {missing}")
    os.replace(staging, output)
    return report
