"""Apply the frozen code-benchmark screen to non-Stack-v3 source successors."""

import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from speck.code_contamination import (
    _dump_line,
    _fingerprint,
    _match_document,
    _ngram_index,
    _sha256,
    _task_payloads,
    _write_json,
    validate_contamination_config,
)
from speck.stack_v3_refine import _sample_partition

FORMAT = "speck_code_contamination_successor"
FORMAT_VERSION = 1
REPORT_FORMAT = "speck_code_contamination_successor_result"


def validate_successor_config(config, *, config_dir=None):
    """Validate a successor while reusing the frozen v1 benchmark-policy contract."""

    if not isinstance(config, dict) or set(config) != {
        "format",
        "format_version",
        "status",
        "input",
        "benchmarks",
        "policy",
        "downstream_partition",
        "output_directory",
    }:
        raise ValueError("code contamination successor has unexpected fields")
    if config["format"] != FORMAT or config["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported code contamination successor format")
    source = config["input"]
    if not isinstance(source, dict) or set(source) != {
        "path",
        "sha256",
        "report",
        "report_sha256",
        "parent_format",
        "parent_status",
    }:
        raise ValueError("successor input has unexpected fields")
    for key in ("parent_format", "parent_status"):
        if not isinstance(source[key], str) or not source[key]:
            raise ValueError(f"input.{key} must be non-empty")

    frozen = {
        **config,
        "format": "speck_code_contamination",
        "input": {
            key: source[key]
            for key in ("path", "sha256", "report", "report_sha256")
        },
    }
    normalized = validate_contamination_config(frozen, config_dir=config_dir)
    normalized["format"] = FORMAT
    normalized["input"] = {
        **normalized["input"],
        "parent_format": source["parent_format"],
        "parent_status": source["parent_status"],
    }
    normalized.pop("plan_fingerprint")
    normalized["plan_fingerprint"] = _fingerprint(normalized)
    return normalized


def load_successor_config(path):
    path = Path(path).resolve()
    return validate_successor_config(
        json.loads(path.read_text(encoding="utf-8")), config_dir=path.parent
    )


def _validated_config(config):
    if "plan_fingerprint" not in config:
        return validate_successor_config(config)
    payload = {key: value for key, value in config.items() if key != "plan_fingerprint"}
    if config["plan_fingerprint"] != _fingerprint(payload):
        raise ValueError("normalized successor plan fingerprint mismatch")
    return config


def _verify(path, digest):
    path = Path(path)
    if not path.is_file() or _sha256(path) != digest:
        raise ValueError(f"code contamination successor identity mismatch: {path}")
    return path


def scan_code_contamination_successor(config, *, restart=False):
    """Remove critical matches from a hash-bound generic code-source successor."""

    config = _validated_config(config)
    input_path = _verify(config["input"]["path"], config["input"]["sha256"])
    parent_path = _verify(config["input"]["report"], config["input"]["report_sha256"])
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    parent_tokenizer = parent.get("outputs", {}).get("tokenizer_input", {})
    if (
        parent.get("format") != config["input"]["parent_format"]
        or parent.get("status") != config["input"]["parent_status"]
        or parent_tokenizer.get("sha256") != config["input"]["sha256"]
        or parent.get("gates", {}).get("training_authority") != "blocked"
    ):
        raise ValueError("code contamination successor parent report is invalid")

    pattern, tasks, benchmarks = _task_payloads(config)
    primary_index = _ngram_index(tasks, config["policy"]["primary_ngram"], config["policy"])
    sensitivity_index = _ngram_index(
        tasks, config["policy"]["sensitivity_ngram"], config["policy"]
    )
    output = Path(config["output_directory"])
    staging = output.with_name(output.name + ".building")
    if output.exists():
        raise FileExistsError(f"code decontamination successor already exists: {output}")
    if staging.exists():
        if not restart:
            raise FileExistsError(f"incomplete code decontamination successor exists: {staging}")
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    cleaned_path = staging / "tokenizer-input.jsonl"
    attribution_path = staging / "attribution.jsonl"
    counts = Counter()
    partitions = Counter()
    removed = []
    sensitivity_records = []
    with (
        input_path.open(encoding="utf-8") as source,
        cleaned_path.open("w", encoding="utf-8") as cleaned,
        attribution_path.open("w", encoding="utf-8") as attribution,
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
            _dump_line(cleaned, record)
            _dump_line(attribution, {key: value for key, value in record.items() if key != "text"})
        for handle in (cleaned, attribution):
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
        "outputs": {
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
        },
        "gates": {
            "parent_input_identity": "pass",
            "benchmark_payload_identity_and_task_count": "pass",
            "frozen_exact_and_ngram_policy": "pass",
            "critical_matches_removed": "pass",
            "downstream_tokenizer_partition_yield": "fail" if missing else "pass",
            "manual_legal_acceptance": "pending",
            "full_corpus_global_deduplication": "pending",
            "acquisition_cleanup_and_resume": "pending",
            "training_authority": "blocked",
        },
    }
    if missing:
        report["missing_downstream_partition_bytes"] = missing
    _write_json(staging / "report.json", report)
    if missing:
        raise RuntimeError(f"successor decontamination did not preserve quota: {missing}")
    os.replace(staging, output)
    return report
