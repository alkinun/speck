"""Validate the blocked freeze of technically qualified tokenizer inputs."""

import hashlib
import json
from pathlib import Path

FORMAT = "speck_tokenizer_inputs_freeze"
FORMAT_VERSION = 1
CATEGORIES = ("web", "code", "math", "synthetic", "science", "reference")


def _sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _exact_keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{name} must contain exactly: {', '.join(sorted(expected))}")


def _resolve(value, config_dir, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(value).expanduser()
    return (config_dir / path).resolve() if not path.is_absolute() else path.resolve()


def validate_tokenizer_inputs_freeze(value, *, config_dir=None, verify_files=False):
    """Require complete 30-source technical identity while preserving every blocker."""

    config_dir = Path(config_dir or ".").resolve()
    _exact_keys(
        value,
        {
            "format",
            "format_version",
            "status",
            "source_registry",
            "source_registry_sha256",
            "categories",
            "targets",
            "rights_template",
            "rights_template_sha256",
            "rights_status",
            "production_operations_status",
            "firewall_status",
            "executable_tokenizer_config",
            "training_authority",
        },
        "tokenizer inputs freeze",
    )
    if (
        value["format"] != FORMAT
        or value["format_version"] != FORMAT_VERSION
        or value["status"] != "technically_frozen_rights_and_production_blocked_not_executable"
        or value["training_authority"] != "blocked"
        or not value["executable_tokenizer_config"].startswith("forbidden")
    ):
        raise ValueError("tokenizer input freeze must remain non-executable and blocked")
    registry_path = _resolve(value["source_registry"], config_dir, "source registry")
    rights_path = _resolve(value["rights_template"], config_dir, "rights template")
    if verify_files and (
        not registry_path.is_file()
        or _sha256(registry_path) != value["source_registry_sha256"]
        or not rights_path.is_file()
        or _sha256(rights_path) != value["rights_template_sha256"]
    ):
        raise ValueError("tokenizer source registry or rights template identity mismatch")
    registry = json.loads(registry_path.read_text())
    allocations = registry["tokenizer_sample_allocations"]
    expected = [allocation["source_id"] for allocation in allocations]
    source_categories = {source["id"]: source["category"] for source in registry["sources"]}
    categories = value["categories"]
    if not isinstance(categories, list) or [category.get("id") for category in categories] != list(
        CATEGORIES
    ):
        raise ValueError("tokenizer categories must use frozen order")
    observed = []
    normalized_categories = []
    for category in categories:
        _exact_keys(
            category,
            {
                "id",
                "technical_result",
                "technical_result_sha256",
                "parent_report",
                "parent_report_sha256",
                "parent_format",
                "parent_status",
                "inputs",
            },
            f"category {category.get('id')}",
        )
        technical_path = _resolve(
            category["technical_result"], config_dir, f"category {category['id']} technical result"
        )
        parent_path = _resolve(
            category["parent_report"], config_dir, f"category {category['id']} parent report"
        )
        if verify_files and (
            not technical_path.is_file()
            or _sha256(technical_path) != category["technical_result_sha256"]
            or not parent_path.is_file()
            or _sha256(parent_path) != category["parent_report_sha256"]
        ):
            raise ValueError(f"category {category['id']} evidence identity mismatch")
        report = json.loads(parent_path.read_text())
        if (
            report.get("format") != category["parent_format"]
            or report.get("status") != category["parent_status"]
            or report.get("gates", {}).get("training_authority") != "blocked"
        ):
            raise ValueError(
                f"category {category['id']} parent report is not blocked technical evidence"
            )
        report_sources = {source["id"]: source for source in report["sources"]}
        normalized_inputs = []
        for item in category["inputs"]:
            _exact_keys(
                item,
                {
                    "source_id",
                    "runtime_source_id",
                    "path",
                    "sha256",
                    "format",
                    "text_column",
                    "training_bytes",
                    "evaluation_bytes",
                },
                "tokenizer input",
            )
            source_id = item["source_id"]
            observed.append(source_id)
            if source_categories.get(source_id) != category["id"]:
                raise ValueError(f"tokenizer input category mismatch: {source_id}")
            allocation = next(
                (allocation for allocation in allocations if allocation["source_id"] == source_id),
                None,
            )
            if allocation is None or any(
                item[key] != allocation[key] for key in ("training_bytes", "evaluation_bytes")
            ):
                raise ValueError(f"tokenizer input allocation mismatch: {source_id}")
            runtime = report_sources.get(item["runtime_source_id"])
            if runtime is None or runtime["outputs"]["tokenizer_input"]["sha256"] != item["sha256"]:
                raise ValueError(f"tokenizer input runtime lineage mismatch: {source_id}")
            path = _resolve(item["path"], config_dir, f"tokenizer input {source_id}")
            if verify_files and (not path.is_file() or _sha256(path) != item["sha256"]):
                raise ValueError(f"tokenizer input file identity mismatch: {source_id}")
            if item["format"] != "jsonl" or item["text_column"] != "text":
                raise ValueError(f"tokenizer input format mismatch: {source_id}")
            normalized_inputs.append({**item, "path": str(path)})
        normalized_categories.append(
            {
                **category,
                "technical_result": str(technical_path),
                "parent_report": str(parent_path),
                "inputs": normalized_inputs,
            }
        )
    if observed != expected or len(observed) != 30:
        raise ValueError("tokenizer inputs do not exactly follow all registry allocations")
    targets = value["targets"]
    expected_targets = {
        "training_bytes_per_category": 100_000_000,
        "evaluation_bytes_per_category": 10_000_000,
        "total_training_bytes": 600_000_000,
        "total_evaluation_bytes": 60_000_000,
    }
    if targets != expected_targets or any(
        sum(item[key] for item in category["inputs"]) != targets[f"{key}_per_category"]
        for category in categories
        for key in ("training_bytes", "evaluation_bytes")
    ):
        raise ValueError("tokenizer input targets are not exactly balanced")
    if (
        value["rights_status"] != "30_pending_0_approved_0_rejected"
        or not value["production_operations_status"].startswith("fixture_tooling_only")
        or not value["firewall_status"].startswith("fixture_tooling_only")
    ):
        raise ValueError("tokenizer input blockers changed unexpectedly")
    return {
        **value,
        "source_registry": str(registry_path),
        "rights_template": str(rights_path),
        "categories": normalized_categories,
    }


def load_tokenizer_inputs_freeze(path, *, verify_files=False):
    path = Path(path).resolve()
    return validate_tokenizer_inputs_freeze(
        json.loads(path.read_text()), config_dir=path.parent, verify_files=verify_files
    )
