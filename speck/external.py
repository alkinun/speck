"""Validate and source-qualify pinned external evaluation suites."""

import hashlib
import json
import re
import subprocess
from pathlib import Path

COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _load_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load external suite config {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"external suite config must contain an object: {path}")
    return value


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(value, keys, context):
    missing = sorted(set(keys) - set(value))
    if missing:
        raise ValueError(f"{context} is missing required fields: {', '.join(missing)}")


def _validate_source(source, context):
    _require(
        source,
        {"id", "repository", "revision", "ref", "license", "required_files"},
        context,
    )
    if not source["repository"].startswith(("https://github.com/", "https://huggingface.co/")):
        raise ValueError(f"{context} repository is not an approved upstream URL")
    if not COMMIT_PATTERN.fullmatch(source["revision"]):
        raise ValueError(f"{context} revision must be a full commit")
    if not source["required_files"]:
        raise ValueError(f"{context} requires source-file pins")
    paths = []
    for entry in source["required_files"]:
        _require(entry, {"path", "sha256"}, f"{context} file pin")
        if Path(entry["path"]).is_absolute() or ".." in Path(entry["path"]).parts:
            raise ValueError(f"{context} file pin must stay inside its checkout")
        if not SHA256_PATTERN.fullmatch(entry["sha256"]):
            raise ValueError(f"{context} file pin has an invalid SHA-256")
        paths.append(entry["path"])
    if len(set(paths)) != len(paths):
        raise ValueError(f"{context} file pins contain duplicate paths")


def validate_external_suite(path):
    """Validate one checked external-suite contract without accessing its checkout."""

    path = Path(path).expanduser().resolve()
    config = _load_json(path)
    _require(
        config,
        {
            "format",
            "format_version",
            "suite_id",
            "suite_version",
            "status",
            "upstream",
            "dependencies",
            "benchmark",
            "data",
            "model_adapter",
            "release_use",
        },
        "external suite",
    )
    if config["format"] != "speck_external_evaluation_suite" or config["format_version"] != 1:
        raise ValueError("external suite must use format version 1")
    if config["suite_id"] not in {"ruler", "nolima", "helmet"}:
        raise ValueError("external suite id is not recognized")
    if "blocked" not in config["status"]:
        raise ValueError("unexecuted external suites must preserve their blocker status")
    sources = [config["upstream"], *config["dependencies"]]
    identifiers = [source.get("id") for source in sources]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("external suite source ids must be unique")
    for index, source in enumerate(sources):
        _validate_source(source, f"external source {index}")
    lengths = config["benchmark"].get("lengths")
    if (
        not isinstance(lengths, list)
        or not lengths
        or lengths != sorted(set(lengths))
        or any(
            isinstance(length, bool) or not isinstance(length, int) or length < 1
            for length in lengths
        )
    ):
        raise ValueError("external suite lengths must be sorted unique positive integers")
    for section in ("data", "model_adapter"):
        if "blocked" not in config[section].get("status", ""):
            raise ValueError(f"external suite {section} must state its unresolved blocker")
    adapter = config["model_adapter"]
    if adapter["status"].startswith("endpoint_protocol_qualified"):
        _require(
            adapter,
            {"qualification", "qualification_sha256", "runner_revision", "settings"},
            "qualified external model adapter",
        )
        if not SHA256_PATTERN.fullmatch(adapter["qualification_sha256"]):
            raise ValueError("external model-adapter qualification has an invalid SHA-256")
        if not COMMIT_PATTERN.fullmatch(adapter["runner_revision"]):
            raise ValueError("external model-adapter qualification has an invalid revision")
        repository_root = path.parents[3]
        qualification_path = repository_root / adapter["qualification"]
        if (
            not qualification_path.is_file()
            or _file_sha256(qualification_path) != adapter["qualification_sha256"]
        ):
            raise ValueError("external model-adapter qualification artifact does not match")
        qualification = _load_json(qualification_path)
        if (
            qualification.get("format") != "speck_evaluation_endpoint_qualification"
            or qualification.get("status")
            != "qualified_for_serialized_openai_correctness_evaluation"
            or qualification.get("runner_revision") != adapter["runner_revision"]
            or config["suite_id"] not in qualification.get("qualified_consumers", {})
            or not qualification.get("export", {}).get("parity", {}).get("passed")
            or qualification.get("export", {}).get("maximum_context", 0) < min(lengths)
        ):
            raise ValueError("external model-adapter qualification artifact is invalid")
    return config


def _git_output(directory, *args):
    return subprocess.run(
        ["git", "-C", str(directory), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def qualify_external_suite(path, checkouts):
    """Verify exact commits and source-file hashes for one external-suite checkout set."""

    config = validate_external_suite(path)
    sources = [config["upstream"], *config["dependencies"]]
    if set(checkouts) != {source["id"] for source in sources}:
        raise ValueError("checkout ids must exactly match the suite's upstream and dependencies")
    qualified = []
    for source in sources:
        directory = Path(checkouts[source["id"]]).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError(f"external checkout does not exist: {directory}")
        revision = _git_output(directory, "rev-parse", "HEAD")
        if revision != source["revision"]:
            raise ValueError(
                f"external checkout {source['id']} is at {revision}, expected {source['revision']}"
            )
        file_results = []
        for entry in source["required_files"]:
            file_path = directory / entry["path"]
            if not file_path.is_file() or _file_sha256(file_path) != entry["sha256"]:
                raise ValueError(
                    f"external checkout {source['id']} file pin failed: {entry['path']}"
                )
            file_results.append(entry)
        qualified.append(
            {
                "id": source["id"],
                "repository": source["repository"],
                "revision": revision,
                "required_files": file_results,
            }
        )
    return {
        "format": "speck_external_suite_source_qualification",
        "format_version": 1,
        "suite_id": config["suite_id"],
        "suite_version": config["suite_version"],
        "config_path": str(Path(path).expanduser().resolve()),
        "config_sha256": _file_sha256(Path(path).expanduser().resolve()),
        "status": "source_qualified",
        "sources": qualified,
        "remaining_data_status": config["data"]["status"],
        "remaining_model_adapter_status": config["model_adapter"]["status"],
    }
