"""Exclude records named by a fully redacted, pinned Gitleaks report."""

import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from speck.stack_v3_refine import _sample_partition

FORMAT = "speck_gitleaks_filter"
FORMAT_VERSION = 1
REPORT_FORMAT = "speck_gitleaks_filter_result"


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


def validate_gitleaks_filter_config(config, *, config_dir=None):
    """Validate a generic immutable Gitleaks exclusion plan."""

    config_dir = Path(config_dir or ".").resolve()
    _exact_keys(
        config,
        {
            "format",
            "format_version",
            "status",
            "input",
            "scanner",
            "downstream_partition",
            "output_directory",
        },
        "Gitleaks filter",
    )
    if config["format"] != FORMAT or config["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported Gitleaks filter format")
    if config["status"] != "exclusion_authorized_not_training_authority":
        raise ValueError("Gitleaks exclusion must remain non-authoritative")
    source = config["input"]
    _exact_keys(
        source,
        {
            "path",
            "sha256",
            "parent_report",
            "parent_report_sha256",
            "parent_format",
            "parent_status",
        },
        "input",
    )
    normalized_input = {
        **source,
        "path": _path(source["path"], "input.path", config_dir),
        "sha256": _digest(source["sha256"], "input.sha256"),
        "parent_report": _path(source["parent_report"], "input.parent_report", config_dir),
        "parent_report_sha256": _digest(
            source["parent_report_sha256"], "input.parent_report_sha256"
        ),
    }
    for key in ("parent_format", "parent_status"):
        if not isinstance(source[key], str) or not source[key]:
            raise ValueError(f"input.{key} must be non-empty")
    scanner = config["scanner"]
    _exact_keys(
        scanner,
        {
            "name",
            "version",
            "official_url",
            "binary",
            "binary_sha256",
            "release_archive_sha256",
            "report",
            "report_sha256",
            "redaction_percent",
        },
        "scanner",
    )
    if scanner["name"] != "gitleaks" or scanner["redaction_percent"] != 100:
        raise ValueError("scanner must be Gitleaks with full redaction")
    normalized_scanner = {
        **scanner,
        "binary": _path(scanner["binary"], "scanner.binary", config_dir),
        "report": _path(scanner["report"], "scanner.report", config_dir),
    }
    for key in ("binary_sha256", "release_archive_sha256", "report_sha256"):
        normalized_scanner[key] = _digest(scanner[key], f"scanner.{key}")
    if not isinstance(scanner["version"], str) or not isinstance(scanner["official_url"], str):
        raise ValueError("scanner version and official URL must be strings")

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
        or any(isinstance(x, bool) or not isinstance(x, int) or not 0 <= x < modulus for x in remainders)
        or len(remainders) == modulus
    ):
        raise ValueError("invalid downstream partition")
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
    normalized = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "status": config["status"],
        "input": normalized_input,
        "scanner": normalized_scanner,
        "downstream_partition": normalized_partition,
        "output_directory": _path(config["output_directory"], "output_directory", config_dir),
    }
    normalized["plan_fingerprint"] = _fingerprint(normalized)
    return normalized


def load_gitleaks_filter_config(path):
    path = Path(path).resolve()
    return validate_gitleaks_filter_config(
        json.loads(path.read_text(encoding="utf-8")), config_dir=path.parent
    )


def _validated_config(config):
    if "plan_fingerprint" not in config:
        return validate_gitleaks_filter_config(config)
    payload = {key: value for key, value in config.items() if key != "plan_fingerprint"}
    if config["plan_fingerprint"] != _fingerprint(payload):
        raise ValueError("normalized Gitleaks filter fingerprint mismatch")
    return config


def _verify(path, digest):
    path = Path(path)
    if not path.is_file() or _sha256(path) != digest:
        raise ValueError(f"Gitleaks filter input identity mismatch: {path}")
    return path


def _blocked_records(config, input_path):
    scanner = config["scanner"]
    _verify(scanner["binary"], scanner["binary_sha256"])
    report_path = _verify(scanner["report"], scanner["report_sha256"])
    findings = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(findings, list):
        raise ValueError("Gitleaks report must be a list")
    lines = set()
    rules = Counter()
    for finding in findings:
        if (
            not isinstance(finding, dict)
            or finding.get("Secret") != "REDACTED"
            or finding.get("Line") not in {"", None}
            or Path(finding.get("File", "")).resolve() != input_path.resolve()
        ):
            raise ValueError("Gitleaks report is not fully redacted or targets another input")
        lines.add(_integer(finding.get("StartLine"), "Gitleaks StartLine", 1))
        rule = finding.get("RuleID")
        if not isinstance(rule, str) or not rule:
            raise ValueError("Gitleaks finding has no rule ID")
        rules[rule] += 1
    blocked = set()
    with input_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line_number in lines:
                record = json.loads(line)
                blocked.add(record["released_content_sha256"])
    if len(blocked) != len(lines):
        raise ValueError("Gitleaks report lines did not map to unique records")
    return blocked, findings, rules


def _dump_line(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def apply_gitleaks_filter(config, *, restart=False):
    """Remove every record named by a fully redacted scanner report."""

    config = _validated_config(config)
    input_path = _verify(config["input"]["path"], config["input"]["sha256"])
    parent_path = _verify(
        config["input"]["parent_report"], config["input"]["parent_report_sha256"]
    )
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if (
        parent.get("format") != config["input"]["parent_format"]
        or parent.get("status") != config["input"]["parent_status"]
        or parent.get("outputs", {}).get("tokenizer_input", {}).get("sha256")
        != config["input"]["sha256"]
        or parent.get("gates", {}).get("training_authority") != "blocked"
    ):
        raise ValueError("Gitleaks filter parent report is invalid")
    blocked, findings, rules = _blocked_records(config, input_path)
    output = Path(config["output_directory"])
    staging = output.with_name(output.name + ".building")
    if output.exists():
        raise FileExistsError(f"Gitleaks-filtered output already exists: {output}")
    if staging.exists():
        if not restart:
            raise FileExistsError(f"incomplete Gitleaks-filtered output exists: {staging}")
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    cleaned_path = staging / "tokenizer-input.jsonl"
    attribution_path = staging / "attribution.jsonl"
    counts = Counter()
    partitions = Counter()
    languages = Counter()
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
                raise ValueError(f"invalid filtered-source record at line {line_number}")
            counts["records_seen"] += 1
            if digest in blocked:
                counts["records_rejected_gitleaks"] += 1
                counts["bytes_rejected_gitleaks"] += len(text.encode())
                continue
            counts["records_retained"] += 1
            size = len(text.encode())
            languages[record.get("language")] += size
            partition = _sample_partition(record, config["downstream_partition"])
            partitions[f"{partition}_records"] += 1
            partitions[f"{partition}_bytes"] += size
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
            "gitleaks_exclusion_incomplete_not_training_authority"
            if missing
            else "gitleaks_exclusion_complete_not_training_authority"
        ),
        "plan_fingerprint": config["plan_fingerprint"],
        "input": config["input"],
        "scanner": {
            **config["scanner"],
            "findings": len(findings),
            "affected_records": len(blocked),
            "rule_counts": dict(sorted(rules.items())),
            "excluded_released_content_sha256": sorted(blocked),
        },
        "counts": dict(sorted(counts.items())),
        "language_bytes": dict(sorted(languages.items())),
        "downstream_partition": {
            **config["downstream_partition"],
            "observed": dict(sorted(partitions.items())),
        },
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
            "scanner_binary_report_and_redaction": "pass",
            "affected_records_excluded": "pass",
            "downstream_tokenizer_partition_yield": "fail" if missing else "pass",
            "cross_source_near_duplicates": "pending",
            "benchmark_contamination": "pending",
            "manual_legal_acceptance": "pending",
            "training_authority": "blocked",
        },
    }
    if missing:
        report["missing_downstream_partition_bytes"] = missing
    _write_json(staging / "report.json", report)
    if missing:
        raise RuntimeError(f"Gitleaks exclusion did not preserve downstream quota: {missing}")
    os.replace(staging, output)
    return report
