"""Decontaminate bounded text sources against immutable evaluation payloads."""

import gzip
import hashlib
import json
import os
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from speck.stack_v3_refine import _sample_partition

FORMAT = "speck_text_contamination"
FORMAT_VERSION = 1
REPORT_FORMAT = "speck_text_contamination_result"
TEXT_CATEGORIES = {"web", "math", "synthetic", "science", "reference"}


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


def _dump_line(handle, value):
    handle.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    )


def _exact_keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{name} must contain exactly: {', '.join(sorted(expected))}")


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _identifier(value, name):
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError(f"{name} must be a non-empty path component")
    return value


def _digest(value, name):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _path(value, name, config_dir):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(value).expanduser()
    return str((config_dir / path).resolve() if not path.is_absolute() else path.resolve())


def validate_text_contamination_config(config, *, config_dir=None):
    """Validate and normalize a frozen multi-source text-contamination plan."""

    config_dir = Path(config_dir or ".").resolve()
    _exact_keys(
        config,
        {
            "format",
            "format_version",
            "status",
            "sources",
            "benchmarks",
            "policy",
            "output_directory",
        },
        "text contamination",
    )
    if config["format"] != FORMAT or config["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported text contamination format")
    if config["status"] != "scan_authorized_not_training_authority":
        raise ValueError("text contamination scan must remain non-authoritative")

    sources = config["sources"]
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources must be a non-empty list")
    normalized_sources = []
    source_ids = []
    categories = set()
    for index, source in enumerate(sources):
        _exact_keys(
            source,
            {
                "id",
                "category",
                "path",
                "sha256",
                "parent_report",
                "parent_report_sha256",
                "parent_format",
                "parent_status",
                "parent_source_id",
                "partition",
            },
            f"source {index}",
        )
        source_id = _identifier(source["id"], f"source {index} id")
        source_ids.append(source_id)
        category = source["category"]
        if category not in TEXT_CATEGORIES:
            raise ValueError(f"source {source_id} has unsupported category")
        categories.add(category)
        for key in ("parent_format", "parent_status", "parent_source_id"):
            if not isinstance(source[key], str) or not source[key]:
                raise ValueError(f"source {source_id} {key} must be non-empty")
        partition = source["partition"]
        _exact_keys(
            partition,
            {
                "seed",
                "modulus",
                "evaluation_remainders",
                "training_bytes",
                "evaluation_bytes",
            },
            f"source {source_id} partition",
        )
        modulus = _integer(partition["modulus"], f"source {source_id} modulus", 2)
        remainders = partition["evaluation_remainders"]
        if (
            not isinstance(remainders, list)
            or not remainders
            or len(remainders) != len(set(remainders))
            or any(
                isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < modulus
                for value in remainders
            )
            or len(remainders) == modulus
        ):
            raise ValueError(f"source {source_id} has invalid evaluation remainders")
        normalized_sources.append(
            {
                **source,
                "path": _path(source["path"], f"source {source_id} path", config_dir),
                "sha256": _digest(source["sha256"], f"source {source_id} sha256"),
                "parent_report": _path(
                    source["parent_report"], f"source {source_id} parent report", config_dir
                ),
                "parent_report_sha256": _digest(
                    source["parent_report_sha256"],
                    f"source {source_id} parent report sha256",
                ),
                "partition": {
                    "seed": _integer(partition["seed"], f"source {source_id} seed"),
                    "modulus": modulus,
                    "evaluation_remainders": sorted(remainders),
                    "training_bytes": _integer(
                        partition["training_bytes"], f"source {source_id} training bytes", 1
                    ),
                    "evaluation_bytes": _integer(
                        partition["evaluation_bytes"], f"source {source_id} evaluation bytes", 1
                    ),
                },
            }
        )
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source IDs must be unique")
    if len(categories) != 1:
        raise ValueError("one contamination plan must contain exactly one category")

    benchmarks = config["benchmarks"]
    if not isinstance(benchmarks, list) or not benchmarks:
        raise ValueError("benchmarks must be a non-empty list")
    normalized_benchmarks = []
    benchmark_ids = []
    for index, benchmark in enumerate(benchmarks):
        _exact_keys(
            benchmark,
            {
                "id",
                "source",
                "revision",
                "official_url",
                "split",
                "path",
                "sha256",
                "format",
                "task_id_field",
                "text_fields",
                "expected_tasks",
            },
            f"benchmark {index}",
        )
        benchmark_id = _identifier(benchmark["id"], f"benchmark {index} id")
        benchmark_ids.append(benchmark_id)
        for key in ("source", "revision", "official_url", "split"):
            if not isinstance(benchmark[key], str) or not benchmark[key]:
                raise ValueError(f"benchmark {benchmark_id} {key} must be non-empty")
        if benchmark["format"] not in {"jsonl", "jsonl_gzip", "parquet"}:
            raise ValueError(f"benchmark {benchmark_id} has unsupported format")
        task_id_field = benchmark["task_id_field"]
        if task_id_field is not None and (not isinstance(task_id_field, str) or not task_id_field):
            raise ValueError(f"benchmark {benchmark_id} task_id_field must be null or a string")
        fields = benchmark["text_fields"]
        if (
            not isinstance(fields, list)
            or not fields
            or any(not isinstance(field, str) or not field for field in fields)
            or len(fields) != len(set(fields))
        ):
            raise ValueError(f"benchmark {benchmark_id} text_fields must be unique strings")
        normalized_benchmarks.append(
            {
                **benchmark,
                "path": _path(benchmark["path"], f"benchmark {benchmark_id} path", config_dir),
                "sha256": _digest(benchmark["sha256"], f"benchmark {benchmark_id} sha256"),
                "text_fields": list(fields),
                "expected_tasks": _integer(
                    benchmark["expected_tasks"], f"benchmark {benchmark_id} expected tasks", 1
                ),
            }
        )
    if len(benchmark_ids) != len(set(benchmark_ids)):
        raise ValueError("benchmark IDs must be unique")

    policy = config["policy"]
    _exact_keys(
        policy,
        {
            "normalization",
            "token_pattern",
            "primary_ngram",
            "sensitivity_ngram",
            "minimum_alphanumeric_tokens",
            "minimum_unique_tokens",
            "maximum_tasks_per_ngram",
            "critical_primary_matches",
            "sensitivity_matches",
            "minimum_exact_field_characters",
            "minimum_exact_field_tokens",
            "exact_anchor_tokens",
        },
        "policy",
    )
    if policy["normalization"] != "NFKC+lower+unicode-word-punctuation":
        raise ValueError("unexpected text contamination normalization")
    if not isinstance(policy["token_pattern"], str) or not policy["token_pattern"]:
        raise ValueError("policy.token_pattern must be non-empty")
    try:
        re.compile(policy["token_pattern"])
    except re.error as error:
        raise ValueError("policy.token_pattern is invalid") from error
    normalized_policy = dict(policy)
    for key, minimum in (
        ("primary_ngram", 2),
        ("sensitivity_ngram", 2),
        ("minimum_alphanumeric_tokens", 1),
        ("minimum_unique_tokens", 1),
        ("maximum_tasks_per_ngram", 1),
        ("critical_primary_matches", 1),
        ("sensitivity_matches", 1),
        ("minimum_exact_field_characters", 1),
        ("minimum_exact_field_tokens", 1),
        ("exact_anchor_tokens", 1),
    ):
        normalized_policy[key] = _integer(policy[key], f"policy.{key}", minimum)
    if normalized_policy["maximum_tasks_per_ngram"] != 1:
        raise ValueError("text contamination requires task-unique n-grams")
    if normalized_policy["sensitivity_ngram"] > normalized_policy["primary_ngram"]:
        raise ValueError("sensitivity n-gram cannot exceed primary n-gram")
    if normalized_policy["exact_anchor_tokens"] > normalized_policy["minimum_exact_field_tokens"]:
        raise ValueError("exact anchor cannot exceed minimum exact-field tokens")

    normalized = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "status": config["status"],
        "sources": normalized_sources,
        "benchmarks": normalized_benchmarks,
        "policy": normalized_policy,
        "output_directory": _path(config["output_directory"], "output directory", config_dir),
    }
    normalized["plan_fingerprint"] = _fingerprint(normalized)
    return normalized


def load_text_contamination_config(path):
    path = Path(path).resolve()
    return validate_text_contamination_config(
        json.loads(path.read_text(encoding="utf-8")), config_dir=path.parent
    )


def _validated_config(config):
    if "plan_fingerprint" not in config:
        return validate_text_contamination_config(config)
    payload = {key: value for key, value in config.items() if key != "plan_fingerprint"}
    if config["plan_fingerprint"] != _fingerprint(payload):
        raise ValueError("normalized text contamination plan fingerprint mismatch")
    return config


def _verify(path, digest, description):
    path = Path(path)
    if not path.is_file() or _sha256(path) != digest:
        raise ValueError(f"{description} identity mismatch: {path}")
    return path


def _verify_source(source):
    path = _verify(source["path"], source["sha256"], f"source {source['id']}")
    report_path = _verify(
        source["parent_report"],
        source["parent_report_sha256"],
        f"source {source['id']} parent report",
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    entries = [
        entry
        for entry in report.get("sources", [])
        if entry.get("id") == source["parent_source_id"]
    ]
    if (
        report.get("format") != source["parent_format"]
        or report.get("status") != source["parent_status"]
        or report.get("gates", {}).get("training_authority") != "blocked"
        or len(entries) != 1
        or entries[0].get("outputs", {}).get("tokenizer_input", {}).get("sha256")
        != source["sha256"]
    ):
        raise ValueError(f"source {source['id']} parent report is invalid")
    return path


def _iter_rows(benchmark):
    path = Path(benchmark["path"])
    if benchmark["format"] in {"jsonl", "jsonl_gzip"}:
        opener = gzip.open if benchmark["format"] == "jsonl_gzip" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    parquet = pq.ParquetFile(path)
    columns = list(benchmark["text_fields"])
    if benchmark["task_id_field"] is not None:
        columns.append(benchmark["task_id_field"])
    for batch in parquet.iter_batches(
        columns=sorted(set(columns)), batch_size=256, use_threads=False
    ):
        yield from batch.to_pylist()


def _flatten_text(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        result = []
        for key in sorted(value):
            result.extend(_flatten_text(value[key]))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_flatten_text(item))
        return result
    if value is None:
        return []
    return [str(value)]


def _tokens(text, pattern):
    return pattern.findall(unicodedata.normalize("NFKC", text).lower())


def _informative(ngram, policy):
    return (
        sum(any(character.isalnum() for character in token) for token in ngram)
        >= policy["minimum_alphanumeric_tokens"]
        and len(set(ngram)) >= policy["minimum_unique_tokens"]
    )


def _load_tasks(config):
    pattern = re.compile(config["policy"]["token_pattern"])
    tasks = {}
    manifest = []
    for benchmark in config["benchmarks"]:
        _verify(benchmark["path"], benchmark["sha256"], f"benchmark {benchmark['id']}")
        count = 0
        for row_index, row in enumerate(_iter_rows(benchmark)):
            task_id = (
                row_index
                if benchmark["task_id_field"] is None
                else row.get(benchmark["task_id_field"])
            )
            if not isinstance(task_id, (str, int)):
                raise ValueError(f"benchmark {benchmark['id']} has an invalid task ID")
            reference = f"{benchmark['id']}:{task_id}"
            if reference in tasks:
                raise ValueError(f"duplicate benchmark task reference: {reference}")
            fields = []
            for name in benchmark["text_fields"]:
                if name not in row:
                    raise ValueError(f"benchmark {benchmark['id']} is missing field {name}")
                for value in _flatten_text(row[name]):
                    tokens = _tokens(value, pattern)
                    if tokens:
                        fields.append(tokens)
            if not fields:
                raise ValueError(f"benchmark task {reference} contains no text")
            tasks[reference] = fields
            count += 1
        if count != benchmark["expected_tasks"]:
            raise ValueError(
                f"benchmark {benchmark['id']} task count {count} != {benchmark['expected_tasks']}"
            )
        manifest.append({**benchmark, "tasks": count})
    return pattern, tasks, manifest


def _ngram_index(tasks, size, policy):
    index = {}
    for reference, fields in tasks.items():
        for tokens in fields:
            for position in range(len(tokens) - size + 1):
                ngram = tuple(tokens[position : position + size])
                if _informative(ngram, policy):
                    owner = index.get(ngram)
                    if owner is None and ngram not in index:
                        index[ngram] = reference
                    elif owner != reference:
                        index[ngram] = None
    return {ngram: reference for ngram, reference in index.items() if reference is not None}


def _exact_index(tasks, policy):
    anchor_size = policy["exact_anchor_tokens"]
    index = defaultdict(list)
    eligible = 0
    for reference, fields in tasks.items():
        for field_index, tokens in enumerate(fields):
            normalized = " ".join(tokens)
            if (
                len(normalized) < policy["minimum_exact_field_characters"]
                or len(tokens) < policy["minimum_exact_field_tokens"]
            ):
                continue
            anchors = (
                tuple(tokens[position : position + anchor_size])
                for position in range(len(tokens) - anchor_size + 1)
                if _informative(tuple(tokens[position : position + anchor_size]), policy)
            )
            anchor = min(anchors, default=None)
            if anchor is None:
                continue
            index[anchor].append((reference, field_index, tuple(tokens)))
            eligible += 1
    return dict(index), eligible


def _contains_tokens(document, field):
    size = len(field)
    if size > len(document):
        return False
    first = field[0]
    return any(
        document[position : position + size] == list(field)
        for position in range(len(document) - size + 1)
        if document[position] == first
    )


def _match_document(text, pattern, primary_index, sensitivity_index, exact_index, policy):
    tokens = _tokens(text, pattern)
    primary = Counter()
    sensitivity = Counter()
    for size, index, counts in (
        (policy["primary_ngram"], primary_index, primary),
        (policy["sensitivity_ngram"], sensitivity_index, sensitivity),
    ):
        observed = set()
        for position in range(len(tokens) - size + 1):
            ngram = tuple(tokens[position : position + size])
            if ngram in observed:
                continue
            observed.add(ngram)
            reference = index.get(ngram)
            if reference is not None:
                counts[reference] += 1

    exact = set()
    exact_fields = defaultdict(list)
    anchor_size = policy["exact_anchor_tokens"]
    candidates = set()
    for position in range(len(tokens) - anchor_size + 1):
        candidates.update(exact_index.get(tuple(tokens[position : position + anchor_size]), ()))
    for reference, field_index, field in candidates:
        if _contains_tokens(tokens, field):
            exact.add(reference)
            exact_fields[reference].append(field_index)

    critical = {
        reference
        for reference, count in primary.items()
        if count >= policy["critical_primary_matches"]
    } | exact
    sensitivity_only = {
        reference
        for reference, count in sensitivity.items()
        if count >= policy["sensitivity_matches"] and reference not in critical
    }
    return critical, sensitivity_only, primary, sensitivity, exact_fields


def scan_text_contamination(config, *, restart=False):
    """Remove critical matches and preserve immutable per-source successors."""

    config = _validated_config(config)
    inputs = [(source, _verify_source(source)) for source in config["sources"]]
    pattern, tasks, benchmarks = _load_tasks(config)
    policy = config["policy"]
    primary_index = _ngram_index(tasks, policy["primary_ngram"], policy)
    sensitivity_index = _ngram_index(tasks, policy["sensitivity_ngram"], policy)
    exact_index, exact_fields = _exact_index(tasks, policy)

    output = Path(config["output_directory"])
    staging = output.with_name(output.name + ".building")
    if output.exists():
        raise FileExistsError(f"text contamination result already exists: {output}")
    if staging.exists():
        if not restart:
            raise FileExistsError(f"incomplete text contamination result exists: {staging}")
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    source_reports = []
    all_pass = True
    total_critical = Counter()
    total_sensitivity = Counter()
    for source, input_path in inputs:
        clean_path = staging / f"{source['id']}.jsonl"
        attribution_path = staging / f"{source['id']}-attribution.jsonl"
        counts = Counter()
        partitions = Counter()
        critical_removed = []
        sensitivity_records = []
        critical_benchmarks = Counter()
        sensitivity_benchmarks = Counter()
        with (
            input_path.open(encoding="utf-8") as handle,
            clean_path.open("w", encoding="utf-8") as cleaned,
            attribution_path.open("w", encoding="utf-8") as attribution,
        ):
            for line_number, line in enumerate(handle, 1):
                record = json.loads(line)
                text = record.get("text")
                digest = record.get("released_content_sha256")
                if (
                    not isinstance(text, str)
                    or not isinstance(digest, str)
                    or hashlib.sha256(text.encode()).hexdigest() != digest
                ):
                    raise ValueError(f"invalid source record at {source['id']}:{line_number}")
                counts["records_seen"] += 1
                critical, sensitivity, primary, sensitive_counts, exact = _match_document(
                    text,
                    pattern,
                    primary_index,
                    sensitivity_index,
                    exact_index,
                    policy,
                )
                if critical:
                    size = len(text.encode())
                    counts["records_removed_critical"] += 1
                    counts["bytes_removed_critical"] += size
                    for reference in critical:
                        benchmark_id = reference.split(":", 1)[0]
                        critical_benchmarks[benchmark_id] += 1
                        total_critical[benchmark_id] += 1
                    critical_removed.append(
                        {
                            "released_content_sha256": digest,
                            "host": record.get("host"),
                            "url": record.get("url"),
                            "critical_tasks": sorted(critical),
                            "exact_field_indices": {
                                reference: sorted(exact[reference]) for reference in sorted(exact)
                            },
                            "primary_match_counts": {
                                reference: primary[reference] for reference in sorted(critical)
                            },
                        }
                    )
                    continue
                if sensitivity:
                    counts["records_sensitivity_only"] += 1
                    for reference in sensitivity:
                        benchmark_id = reference.split(":", 1)[0]
                        sensitivity_benchmarks[benchmark_id] += 1
                        total_sensitivity[benchmark_id] += 1
                    sensitivity_records.append(
                        {
                            "released_content_sha256": digest,
                            "tasks": sorted(sensitivity),
                            "match_counts": {
                                reference: sensitive_counts[reference]
                                for reference in sorted(sensitivity)
                            },
                        }
                    )
                counts["records_retained"] += 1
                size = len(text.encode())
                partition = _sample_partition(
                    record,
                    {
                        "seed": source["partition"]["seed"],
                        "category": source["category"],
                        "modulus": source["partition"]["modulus"],
                        "evaluation_remainders": source["partition"]["evaluation_remainders"],
                    },
                )
                partitions[f"{partition}_bytes"] += size
                partitions[f"{partition}_records"] += 1
                _dump_line(cleaned, record)
                _dump_line(
                    attribution, {key: value for key, value in record.items() if key != "text"}
                )
            for output_handle in (cleaned, attribution):
                output_handle.flush()
                os.fsync(output_handle.fileno())

        targets = {
            "train_bytes": source["partition"]["training_bytes"],
            "eval_bytes": source["partition"]["evaluation_bytes"],
        }
        missing = {
            split: {"bytes": partitions[split], "target_bytes": target}
            for split, target in targets.items()
            if partitions[split] < target
        }
        if missing:
            all_pass = False
        source_reports.append(
            {
                "id": source["id"],
                "input": source,
                "counts": dict(sorted(counts.items())),
                "critical_by_benchmark": dict(sorted(critical_benchmarks.items())),
                "sensitivity_by_benchmark": dict(sorted(sensitivity_benchmarks.items())),
                "critical_removed": critical_removed,
                "sensitivity_only": sensitivity_records,
                "partition": {**dict(sorted(partitions.items())), "targets": targets},
                "missing": missing,
                "outputs": {
                    "tokenizer_input": {
                        "path": clean_path.name,
                        "bytes": clean_path.stat().st_size,
                        "sha256": _sha256(clean_path),
                    },
                    "attribution": {
                        "path": attribution_path.name,
                        "bytes": attribution_path.stat().st_size,
                        "sha256": _sha256(attribution_path),
                    },
                },
            }
        )

    report = {
        "format": REPORT_FORMAT,
        "format_version": FORMAT_VERSION,
        "status": (
            "decontamination_complete_not_training_authority"
            if all_pass
            else "decontamination_incomplete_not_training_authority"
        ),
        "plan_fingerprint": config["plan_fingerprint"],
        "benchmarks": benchmarks,
        "policy": policy,
        "index": {
            "tasks": len(tasks),
            "primary_unique_ngrams": len(primary_index),
            "sensitivity_unique_ngrams": len(sensitivity_index),
            "exact_fields": exact_fields,
            "exact_anchors": len(exact_index),
        },
        "critical_by_benchmark": dict(sorted(total_critical.items())),
        "sensitivity_by_benchmark": dict(sorted(total_sensitivity.items())),
        "sources": source_reports,
        "gates": {
            "source_and_parent_identity": "pass",
            "benchmark_payload_identity_and_task_count": "pass",
            "frozen_exact_and_ngram_policy": "pass",
            "critical_matches_removed": "pass",
            "post_decontamination_partition_yield": "pass" if all_pass else "fail",
            "manual_legal_acceptance": "pending",
            "production_global_deduplication": "pending",
            "acquisition_cleanup_and_resume": "pending",
            "training_authority": "blocked",
        },
    }
    _write_json(staging / "report.json", report)
    if not all_pass:
        raise RuntimeError("text decontamination violated a source partition quota")
    os.replace(staging, output)
    return report
