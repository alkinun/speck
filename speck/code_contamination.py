"""Scan a code corpus against immutable code-evaluation payloads."""

import gzip
import hashlib
import json
import os
import shutil
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from speck.stack_v3_refine import _sample_partition

FORMAT = "speck_code_contamination"
FORMAT_VERSION = 1
REPORT_FORMAT = "speck_code_contamination_result"


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


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


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


def validate_contamination_config(config, *, config_dir=None):
    """Validate and normalize a frozen code-contamination plan."""

    config_dir = Path(config_dir or ".").resolve()
    _exact_keys(
        config,
        {
            "format",
            "format_version",
            "status",
            "input",
            "benchmarks",
            "policy",
            "downstream_partition",
            "output_directory",
        },
        "code contamination",
    )
    if config["format"] != FORMAT or config["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported code contamination format")
    if config["status"] != "scan_authorized_not_training_authority":
        raise ValueError("code contamination scan must remain non-authoritative")
    source = config["input"]
    _exact_keys(source, {"path", "sha256", "report", "report_sha256"}, "input")
    normalized_input = {
        "path": _path(source["path"], "input.path", config_dir),
        "sha256": _digest(source["sha256"], "input.sha256"),
        "report": _path(source["report"], "input.report", config_dir),
        "report_sha256": _digest(source["report_sha256"], "input.report_sha256"),
    }

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
                "path",
                "sha256",
                "format",
                "task_id_field",
                "text_fields",
                "expected_tasks",
            },
            f"benchmark {index}",
        )
        benchmark_id = benchmark["id"]
        if not isinstance(benchmark_id, str) or not benchmark_id or Path(benchmark_id).name != benchmark_id:
            raise ValueError(f"benchmark {index} id must be a path component")
        benchmark_ids.append(benchmark_id)
        if benchmark["format"] not in {"jsonl", "jsonl_gzip", "parquet"}:
            raise ValueError(f"benchmark {benchmark_id} has unsupported format")
        for key in ("source", "revision", "official_url", "task_id_field"):
            if not isinstance(benchmark[key], str) or not benchmark[key]:
                raise ValueError(f"benchmark {benchmark_id} {key} must be non-empty")
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
                    benchmark["expected_tasks"], f"benchmark {benchmark_id} expected_tasks", 1
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
        },
        "policy",
    )
    if policy["normalization"] != "NFKC+lower+lexical-code-tokens":
        raise ValueError("unexpected contamination normalization")
    if not isinstance(policy["token_pattern"], str) or not policy["token_pattern"]:
        raise ValueError("policy.token_pattern must be non-empty")
    try:
        import re

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
    ):
        normalized_policy[key] = _integer(policy[key], f"policy.{key}", minimum)
    if normalized_policy["sensitivity_ngram"] > normalized_policy["primary_ngram"]:
        raise ValueError("sensitivity n-gram cannot exceed primary n-gram")

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
    if partition["category"] != "code":
        raise ValueError("downstream category must be code")
    modulus = _integer(partition["modulus"], "partition.modulus", 2)
    remainders = partition["evaluation_remainders"]
    if (
        not isinstance(remainders, list)
        or not remainders
        or len(remainders) != len(set(remainders))
        or any(isinstance(x, bool) or not isinstance(x, int) or not 0 <= x < modulus for x in remainders)
        or len(remainders) == modulus
    ):
        raise ValueError("invalid downstream evaluation remainders")
    normalized_partition = {
        **partition,
        "seed": _integer(partition["seed"], "partition.seed"),
        "modulus": modulus,
        "evaluation_remainders": sorted(remainders),
        "training_bytes": _integer(partition["training_bytes"], "partition.training_bytes"),
        "evaluation_bytes": _integer(
            partition["evaluation_bytes"], "partition.evaluation_bytes"
        ),
    }
    if normalized_partition["training_bytes"] + normalized_partition["evaluation_bytes"] < 1:
        raise ValueError("downstream partition must request at least one byte")

    normalized = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "status": config["status"],
        "input": normalized_input,
        "benchmarks": normalized_benchmarks,
        "policy": normalized_policy,
        "downstream_partition": normalized_partition,
        "output_directory": _path(config["output_directory"], "output_directory", config_dir),
    }
    normalized["plan_fingerprint"] = _fingerprint(normalized)
    return normalized


def load_contamination_config(path):
    path = Path(path).resolve()
    return validate_contamination_config(
        json.loads(path.read_text(encoding="utf-8")), config_dir=path.parent
    )


def _validated_config(config):
    if "plan_fingerprint" not in config:
        return validate_contamination_config(config)
    payload = {key: value for key, value in config.items() if key != "plan_fingerprint"}
    if config["plan_fingerprint"] != _fingerprint(payload):
        raise ValueError("normalized contamination plan fingerprint mismatch")
    return config


def _verify(path, digest):
    path = Path(path)
    if not path.is_file() or _sha256(path) != digest:
        raise ValueError(f"code contamination input identity mismatch: {path}")
    return path


def _iter_benchmark_rows(benchmark):
    path = Path(benchmark["path"])
    if benchmark["format"] in {"jsonl", "jsonl_gzip"}:
        opener = gzip.open if benchmark["format"] == "jsonl_gzip" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    parquet = pq.ParquetFile(path)
    columns = [benchmark["task_id_field"], *benchmark["text_fields"]]
    for batch in parquet.iter_batches(columns=columns, batch_size=256, use_threads=False):
        yield from batch.to_pylist()


def _flatten_text(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
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
    alphanumeric = sum(any(character.isalnum() for character in token) for token in ngram)
    return (
        alphanumeric >= policy["minimum_alphanumeric_tokens"]
        and len(set(ngram)) >= policy["minimum_unique_tokens"]
    )


def _task_payloads(config):
    import re

    pattern = re.compile(config["policy"]["token_pattern"])
    tasks = {}
    benchmark_manifest = []
    for benchmark in config["benchmarks"]:
        _verify(benchmark["path"], benchmark["sha256"])
        count = 0
        for row in _iter_benchmark_rows(benchmark):
            task_id = row.get(benchmark["task_id_field"])
            if not isinstance(task_id, (str, int)):
                raise ValueError(f"benchmark {benchmark['id']} has an invalid task ID")
            reference = f"{benchmark['id']}:{task_id}"
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
        benchmark_manifest.append(
            {
                **benchmark,
                "tasks": count,
            }
        )
    return pattern, tasks, benchmark_manifest


def _ngram_index(tasks, size, policy):
    index = defaultdict(set)
    for reference, fields in tasks.items():
        for tokens in fields:
            for position in range(len(tokens) - size + 1):
                ngram = tuple(tokens[position : position + size])
                if _informative(ngram, policy):
                    index[ngram].add(reference)
    maximum = policy["maximum_tasks_per_ngram"]
    return {ngram: next(iter(refs)) for ngram, refs in index.items() if len(refs) <= maximum}


def _match_document(text, pattern, tasks, primary_index, sensitivity_index, policy):
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
    candidates = set(primary) | set(sensitivity)
    joined = " ".join(tokens)
    exact = set()
    for reference in candidates:
        for field in tasks[reference]:
            field_text = " ".join(field)
            if (
                len(field_text) >= policy["minimum_exact_field_characters"]
                and field_text in joined
            ):
                exact.add(reference)
                break
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
    return critical, sensitivity_only, primary, sensitivity, exact


def _dump_line(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def scan_code_contamination(config, *, restart=False):
    """Remove critical benchmark matches and preserve a non-authoritative overlap report."""

    config = _validated_config(config)
    input_path = _verify(config["input"]["path"], config["input"]["sha256"])
    parent_report_path = _verify(
        config["input"]["report"], config["input"]["report_sha256"]
    )
    parent_report = json.loads(parent_report_path.read_text(encoding="utf-8"))
    parent_tokenizer = parent_report.get("outputs", {}).get("tokenizer_input", {})
    if (
        parent_report.get("format") != "speck_stack_v3_refinement_result"
        or parent_report.get("status") != "refinement_complete_not_training_authority"
        or parent_tokenizer.get("sha256") != config["input"]["sha256"]
        or parent_report.get("gates", {}).get("training_authority") != "blocked"
    ):
        raise ValueError("code contamination parent report is invalid")
    pattern, tasks, benchmarks = _task_payloads(config)
    primary_index = _ngram_index(tasks, config["policy"]["primary_ngram"], config["policy"])
    sensitivity_index = _ngram_index(
        tasks, config["policy"]["sensitivity_ngram"], config["policy"]
    )

    output = Path(config["output_directory"])
    staging = output.with_name(output.name + ".building")
    if output.exists():
        raise FileExistsError(f"code decontamination already exists: {output}")
    if staging.exists():
        if not restart:
            raise FileExistsError(f"incomplete code decontamination exists: {staging}")
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    cleaned_path = staging / "tokenizer-input.jsonl"
    attribution_path = staging / "attribution.jsonl"
    repository_path = staging / "repository-sample.jsonl"
    counts = Counter()
    removed = []
    sensitivity_records = []
    partitions = Counter()
    current_repo = None
    current_files = []
    current_metadata = None

    def flush_repository(handle):
        if current_repo is not None and current_files:
            _dump_line(
                handle,
                {
                    "repo_path": current_repo,
                    "repo_id": current_metadata["repo_id"],
                    "commit_id": current_metadata["commit_id"],
                    "files": current_files,
                },
            )

    with (
        input_path.open(encoding="utf-8") as source,
        cleaned_path.open("w", encoding="utf-8") as cleaned,
        attribution_path.open("w", encoding="utf-8") as attribution,
        repository_path.open("w", encoding="utf-8") as repositories,
    ):
        for line_number, line in enumerate(source, 1):
            record = json.loads(line)
            text = record.get("text")
            digest = record.get("released_content_sha256")
            if (
                not isinstance(text, str)
                or not isinstance(digest, str)
                or hashlib.sha256(text.encode()).hexdigest() != digest
            ):
                raise ValueError(f"invalid code record at line {line_number}")
            counts["records_seen"] += 1
            critical, sensitivity_only, primary, sensitivity, exact = _match_document(
                text, pattern, tasks, primary_index, sensitivity_index, config["policy"]
            )
            if critical:
                counts["records_removed_critical"] += 1
                counts["bytes_removed_critical"] += len(text.encode())
                removed.append(
                    {
                        "released_content_sha256": digest,
                        "repo_path": record.get("repo_path"),
                        "file_path": record.get("file_path"),
                        "language": record.get("language"),
                        "critical_tasks": sorted(critical),
                        "exact_field_tasks": sorted(exact),
                        "primary_match_counts": {
                            reference: primary[reference] for reference in sorted(critical)
                        },
                    }
                )
                continue
            if sensitivity_only:
                counts["records_sensitivity_only"] += 1
                sensitivity_records.append(
                    {
                        "released_content_sha256": digest,
                        "tasks": sorted(sensitivity_only),
                        "match_counts": {
                            reference: sensitivity[reference]
                            for reference in sorted(sensitivity_only)
                        },
                    }
                )
            counts["records_retained"] += 1
            size = len(text.encode())
            partition = _sample_partition(record, config["downstream_partition"])
            partitions[f"{partition}_bytes"] += size
            partitions[f"{partition}_records"] += 1
            repo_path = record.get("repo_path")
            if repo_path != current_repo:
                flush_repository(repositories)
                current_repo = repo_path
                current_files = []
                current_metadata = {
                    "repo_id": record.get("repo_id"),
                    "commit_id": record.get("commit_id"),
                }
            _dump_line(cleaned, record)
            _dump_line(attribution, {key: value for key, value in record.items() if key != "text"})
            current_files.append(record)
        flush_repository(repositories)
        for handle in (cleaned, attribution, repositories):
            handle.flush()
            os.fsync(handle.fileno())

    targets = {
        "train_bytes": config["downstream_partition"]["training_bytes"],
        "eval_bytes": config["downstream_partition"]["evaluation_bytes"],
    }
    missing = {
        split: {"bytes": partitions[split], "target_bytes": target}
        for split, target in targets.items()
        if partitions[split] < target
    }
    outputs = {
        "tokenizer_input": {
            "path": cleaned_path.name,
            "bytes": cleaned_path.stat().st_size,
            "sha256": _sha256(cleaned_path),
        },
        "attribution": {
            "path": attribution_path.name,
            "bytes": attribution_path.stat().st_size,
            "sha256": _sha256(attribution_path),
        },
        "repository_sample": {
            "path": repository_path.name,
            "bytes": repository_path.stat().st_size,
            "sha256": _sha256(repository_path),
        },
    }
    report = {
        "format": REPORT_FORMAT,
        "format_version": FORMAT_VERSION,
        "status": (
            "decontamination_incomplete_not_training_authority"
            if missing
            else "decontamination_complete_not_training_authority"
        ),
        "plan_fingerprint": config["plan_fingerprint"],
        "input": config["input"],
        "benchmarks": benchmarks,
        "policy": config["policy"],
        "index": {
            "tasks": len(tasks),
            "primary_unique_ngrams": len(primary_index),
            "sensitivity_unique_ngrams": len(sensitivity_index),
        },
        "counts": dict(sorted(counts.items())),
        "downstream_partition": {
            **config["downstream_partition"],
            "observed": dict(sorted(partitions.items())),
        },
        "critical_removed": removed,
        "sensitivity_only": sensitivity_records,
        "outputs": outputs,
        "gates": {
            "benchmark_payload_identity_and_task_count": "pass",
            "frozen_exact_and_ngram_policy": "pass",
            "critical_matches_removed": "pass",
            "downstream_tokenizer_partition_yield": "fail" if missing else "pass",
            "manual_legal_acceptance": "pending",
            "near_duplicates_cross_source": "pending",
            "acquisition_cleanup_and_resume": "pending",
            "training_authority": "blocked",
        },
    }
    if missing:
        report["missing_downstream_partition_bytes"] = missing
    _write_json(staging / "report.json", report)
    if missing:
        raise RuntimeError(f"decontamination did not preserve downstream quota: {missing}")
    os.replace(staging, output)
    return report
