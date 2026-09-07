"""Qualify a bounded, repository-aware sample from The Stack v3."""

import hashlib
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

FORMAT = "speck_stack_v3_qualification"
FORMAT_VERSION = 1
REPORT_FORMAT = "speck_stack_v3_qualification_result"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_REQUIRED_COLUMNS = {
    "repo_path",
    "repo_id",
    "commit_id",
    "github_metadata",
    "num_files",
    "files",
}
_PII_PLACEHOLDERS = ("<EMAIL>", "<KEY>", "<NAME>", "<PASSWORD>")
_SECRET_PATTERNS = {
    "pem_private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "github_token": re.compile(r"\bgh(?:p|o|u|s|r)_[A-Za-z0-9]{30,255}\b"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
}


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


def _identifier(value, name):
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError(f"{name} must be a single non-empty path component")
    return value


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _exact_keys(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{name} must contain exactly: {', '.join(sorted(keys))}")


def validate_stack_v3_config(config, *, config_dir=None):
    """Validate and normalize a bounded Stack v3 qualification configuration."""

    config_dir = Path(config_dir or ".").resolve()
    _exact_keys(
        config,
        {"format", "format_version", "status", "source", "filters", "output"},
        "Stack v3 qualification",
    )
    if config["format"] != FORMAT or config["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported Stack v3 qualification format")
    if config["status"] != "bounded_profile_authorized_not_training_authority":
        raise ValueError("Stack v3 qualification must remain non-authoritative")

    source = config["source"]
    _exact_keys(source, {"repo", "revision", "release", "files"}, "source")
    if not isinstance(source["repo"], str) or not source["repo"]:
        raise ValueError("source.repo must be non-empty")
    if not isinstance(source["revision"], str) or not _COMMIT.fullmatch(source["revision"]):
        raise ValueError("source.revision must be a 40-character commit")
    if not isinstance(source["release"], str) or not source["release"]:
        raise ValueError("source.release must be non-empty")
    files = source["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("source.files must be a non-empty list")
    normalized_files = []
    names = []
    for index, item in enumerate(files):
        _exact_keys(item, {"path", "size", "sha256"}, f"source file {index}")
        path = item["path"]
        if (
            not isinstance(path, str)
            or not path.startswith("data/")
            or not path.endswith(".parquet")
            or Path(path).is_absolute()
            or ".." in Path(path).parts
        ):
            raise ValueError(f"source file {index} has invalid path")
        names.append(path)
        size = _integer(item["size"], f"source file {path} size", 1)
        if not isinstance(item["sha256"], str) or not _SHA256.fullmatch(item["sha256"]):
            raise ValueError(f"source file {path} sha256 must be lowercase hexadecimal")
        normalized_files.append({"path": path, "size": size, "sha256": item["sha256"]})
    if len(names) != len(set(names)):
        raise ValueError("source file paths must be unique")

    filters = config["filters"]
    _exact_keys(
        filters,
        {
            "license_type",
            "require_detected_licenses",
            "rejected_license_prefixes",
            "exclude_vendor",
            "exclude_forks",
            "min_file_bytes",
            "max_file_bytes",
            "max_sample_bytes_per_language_per_repository",
            "language_sample_bytes",
        },
        "filters",
    )
    if filters["license_type"] != "permissive":
        raise ValueError("only permissive Stack v3 files may be qualified")
    for key in ("require_detected_licenses", "exclude_vendor", "exclude_forks"):
        if not isinstance(filters[key], bool):
            raise ValueError(f"filters.{key} must be boolean")
    prefixes = filters["rejected_license_prefixes"]
    if not isinstance(prefixes, list) or any(not isinstance(value, str) for value in prefixes):
        raise ValueError("rejected_license_prefixes must be a string list")
    minimum = _integer(filters["min_file_bytes"], "filters.min_file_bytes", 1)
    maximum = _integer(filters["max_file_bytes"], "filters.max_file_bytes", 1)
    if minimum > maximum:
        raise ValueError("minimum file bytes cannot exceed maximum")
    repository_cap = _integer(
        filters["max_sample_bytes_per_language_per_repository"],
        "filters.max_sample_bytes_per_language_per_repository",
        1,
    )
    languages = filters["language_sample_bytes"]
    if not isinstance(languages, dict) or not languages:
        raise ValueError("language_sample_bytes must be a non-empty object")
    normalized_languages = {}
    for language, size in languages.items():
        if not isinstance(language, str) or not language:
            raise ValueError("language names must be non-empty strings")
        normalized_languages[language] = _integer(size, f"language {language} sample bytes", 1)

    output = config["output"]
    _exact_keys(output, {"directory", "download_directory", "keep_downloads"}, "output")
    if not isinstance(output["keep_downloads"], bool):
        raise ValueError("output.keep_downloads must be boolean")
    normalized_output = {}
    for key in ("directory", "download_directory"):
        if not isinstance(output[key], str) or not output[key]:
            raise ValueError(f"output.{key} must be a non-empty path")
        path = Path(output[key]).expanduser()
        normalized_output[key] = str(
            (config_dir / path).resolve() if not path.is_absolute() else path.resolve()
        )
    normalized_output["keep_downloads"] = output["keep_downloads"]

    normalized = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "status": config["status"],
        "source": {**source, "files": normalized_files},
        "filters": {
            **filters,
            "rejected_license_prefixes": list(prefixes),
            "min_file_bytes": minimum,
            "max_file_bytes": maximum,
            "max_sample_bytes_per_language_per_repository": repository_cap,
            "language_sample_bytes": normalized_languages,
        },
        "output": normalized_output,
    }
    normalized["plan_fingerprint"] = _fingerprint(normalized)
    return normalized


def load_stack_v3_config(path):
    path = Path(path).resolve()
    return validate_stack_v3_config(
        json.loads(path.read_text(encoding="utf-8")), config_dir=path.parent
    )


def _validated_config(config):
    if "plan_fingerprint" not in config:
        return validate_stack_v3_config(config)
    payload = {key: value for key, value in config.items() if key != "plan_fingerprint"}
    if config["plan_fingerprint"] != _fingerprint(payload):
        raise ValueError("normalized Stack v3 qualification fingerprint mismatch")
    return config


def acquire_stack_v3_files(config):
    """Download only the explicitly declared revision-pinned source files."""

    config = _validated_config(config)
    directory = Path(config["output"]["download_directory"])
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for item in config["source"]["files"]:
        path = directory / item["path"]
        if not path.is_file():
            path = Path(
                hf_hub_download(
                    repo_id=config["source"]["repo"],
                    repo_type="dataset",
                    revision=config["source"]["revision"],
                    filename=item["path"],
                    local_dir=directory,
                )
            )
        if path.stat().st_size != item["size"] or _sha256(path) != item["sha256"]:
            raise ValueError(f"downloaded Stack v3 file identity mismatch: {path}")
        paths.append(path)
    return paths


def _validate_schema(path):
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    missing = _REQUIRED_COLUMNS - set(schema.names)
    if missing:
        raise ValueError(f"Stack v3 shard is missing columns {sorted(missing)}: {path}")
    if not (
        pa.types.is_list(schema.field("files").type)
        or pa.types.is_large_list(schema.field("files").type)
    ):
        raise ValueError(f"Stack v3 files column must be a list: {path}")
    value_type = schema.field("files").type.value_type
    required_file_fields = {
        "content_id",
        "content",
        "size_bytes",
        "file_path",
        "language",
        "is_vendor",
        "license_type",
        "detected_licenses",
    }
    if not pa.types.is_struct(value_type) or required_file_fields - set(value_type.names):
        raise ValueError(f"Stack v3 files struct has an incompatible schema: {path}")
    return parquet


def _secret_counts(text):
    return {name: len(pattern.findall(text)) for name, pattern in _SECRET_PATTERNS.items()}


def _dump_line(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def qualify_stack_v3(config, *, local_files=None, restart=False):
    """Profile and extract a non-authoritative restricted Stack v3 sample."""

    config = _validated_config(config)
    source_files = list(local_files) if local_files is not None else acquire_stack_v3_files(config)
    if len(source_files) != len(config["source"]["files"]):
        raise ValueError("local Stack v3 file count does not match the plan")
    verified_files = []
    for declaration, path in zip(config["source"]["files"], source_files):
        path = Path(path)
        if (
            not path.is_file()
            or path.stat().st_size != declaration["size"]
            or _sha256(path) != declaration["sha256"]
        ):
            raise ValueError(f"local Stack v3 file identity mismatch: {path}")
        verified_files.append(path)

    output = Path(config["output"]["directory"])
    staging = output.with_name(output.name + ".building")
    if output.exists():
        raise FileExistsError(f"Stack v3 qualification already exists: {output}")
    if staging.exists():
        if not restart:
            raise FileExistsError(f"incomplete Stack v3 qualification exists: {staging}")
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    tokenizer_path = staging / "tokenizer-input.jsonl"
    repository_path = staging / "repository-sample.jsonl"
    attribution_path = staging / "attribution.jsonl"
    targets = dict(config["filters"]["language_sample_bytes"])
    sampled_bytes = Counter()
    sample_repositories = set()
    counts = Counter()
    languages = Counter()
    language_bytes = Counter()
    licenses = Counter()
    pii = Counter()
    secrets = Counter()
    released_content_hashes = set()
    source_identity = []
    try:
        with (
            tokenizer_path.open("w", encoding="utf-8") as tokenizer_output,
            repository_path.open("w", encoding="utf-8") as repository_output,
            attribution_path.open("w", encoding="utf-8") as attribution_output,
        ):
            for declaration, path in zip(config["source"]["files"], verified_files):
                parquet = _validate_schema(path)
                source_identity.append(
                    {
                        **declaration,
                        "local_path": str(path.resolve()),
                        "row_groups": parquet.num_row_groups,
                        "rows": parquet.metadata.num_rows,
                    }
                )
                for batch in parquet.iter_batches(
                    columns=sorted(_REQUIRED_COLUMNS), batch_size=64, use_threads=False
                ):
                    for repository in batch.to_pylist():
                        counts["repositories_seen"] += 1
                        metadata = repository.get("github_metadata") or {}
                        if config["filters"]["exclude_forks"] and metadata.get("is_fork"):
                            counts["repositories_rejected_fork"] += 1
                            continue
                        repo_path = repository.get("repo_path")
                        commit_id = repository.get("commit_id")
                        if not isinstance(repo_path, str) or not isinstance(commit_id, str):
                            counts["repositories_rejected_identity"] += 1
                            continue
                        counts["repositories_accepted_for_file_filter"] += 1
                        selected = []
                        per_repo_language_bytes = Counter()
                        files = repository.get("files") or []
                        counts["files_declared_by_repository"] += repository.get("num_files") or 0
                        for file in files:
                            counts["files_seen"] += 1
                            content = file.get("content")
                            content_id = file.get("content_id")
                            if not isinstance(content, str) or not content or not isinstance(
                                content_id, str
                            ):
                                counts["files_rejected_content"] += 1
                                continue
                            encoded = content.encode("utf-8")
                            size = len(encoded)
                            if file.get("size_bytes") != size:
                                counts["files_rejected_size_mismatch"] += 1
                                continue
                            if not _COMMIT.fullmatch(content_id):
                                counts["files_rejected_content_id_format"] += 1
                                continue
                            if hashlib.sha1(encoded).hexdigest() != content_id:
                                counts["files_upstream_content_id_not_plain_sha1"] += 1
                            if config["filters"]["exclude_vendor"] and file.get("is_vendor"):
                                counts["files_rejected_vendor"] += 1
                                continue
                            if file.get("license_type") != config["filters"]["license_type"]:
                                counts["files_rejected_license_type"] += 1
                                continue
                            detected = file.get("detected_licenses") or []
                            if not isinstance(detected, list) or any(
                                not isinstance(license_id, str) for license_id in detected
                            ):
                                counts["files_rejected_detected_license_format"] += 1
                                continue
                            if config["filters"]["require_detected_licenses"] and not detected:
                                counts["files_rejected_missing_detected_license"] += 1
                                continue
                            if any(
                                license_id.startswith(prefix)
                                for license_id in detected
                                for prefix in config["filters"]["rejected_license_prefixes"]
                            ):
                                counts["files_rejected_license_reference"] += 1
                                continue
                            language = file.get("language")
                            if language not in targets:
                                counts["files_rejected_language"] += 1
                                continue
                            if not config["filters"]["min_file_bytes"] <= size <= config["filters"][
                                "max_file_bytes"
                            ]:
                                counts["files_rejected_size"] += 1
                                continue
                            secret_counts = _secret_counts(content)
                            secrets.update(secret_counts)
                            if sum(secret_counts.values()):
                                counts["files_rejected_high_confidence_secret"] += 1
                                continue
                            released_content_sha256 = hashlib.sha256(encoded).hexdigest()
                            content_key = int(released_content_sha256, 16)
                            if content_key in released_content_hashes:
                                counts["files_rejected_exact_duplicate"] += 1
                                continue
                            released_content_hashes.add(content_key)
                            counts["files_accepted"] += 1
                            languages[language] += 1
                            language_bytes[language] += size
                            for license_id in detected:
                                licenses[license_id] += 1
                            for placeholder in _PII_PLACEHOLDERS:
                                pii[placeholder] += content.count(placeholder)
                            remaining = targets[language] - sampled_bytes[language]
                            repo_remaining = (
                                config["filters"][
                                    "max_sample_bytes_per_language_per_repository"
                                ]
                                - per_repo_language_bytes[language]
                            )
                            if remaining <= 0 or repo_remaining <= 0:
                                counts["files_accepted_not_sampled_quota"] += 1
                                continue
                            if size > repo_remaining:
                                counts["files_accepted_not_sampled_repo_cap"] += 1
                                continue
                            record = {
                                "text": content,
                                "content_id": content_id,
                                "released_content_sha256": released_content_sha256,
                                "repo_path": repo_path,
                                "repo_id": repository.get("repo_id"),
                                "commit_id": commit_id,
                                "file_path": file.get("file_path"),
                                "language": language,
                                "detected_licenses": detected,
                                "size_bytes": size,
                            }
                            _dump_line(tokenizer_output, record)
                            _dump_line(
                                attribution_output,
                                {key: value for key, value in record.items() if key != "text"},
                            )
                            selected.append(record)
                            sampled_bytes[language] += size
                            per_repo_language_bytes[language] += size
                            counts["files_sampled"] += 1
                        if selected:
                            sample_repositories.add(repo_path)
                            _dump_line(
                                repository_output,
                                {
                                    "repo_path": repo_path,
                                    "repo_id": repository.get("repo_id"),
                                    "commit_id": commit_id,
                                    "github_metadata": metadata,
                                    "files": selected,
                                },
                            )
            for handle in (tokenizer_output, repository_output, attribution_output):
                handle.flush()
                os.fsync(handle.fileno())

        missing = {
            language: {"sampled_bytes": sampled_bytes[language], "target_bytes": target}
            for language, target in targets.items()
            if sampled_bytes[language] < target
        }
        outputs = {
            "tokenizer_input": {
                "path": tokenizer_path.name,
                "bytes": tokenizer_path.stat().st_size,
                "sha256": _sha256(tokenizer_path),
            },
            "repository_sample": {
                "path": repository_path.name,
                "bytes": repository_path.stat().st_size,
                "sha256": _sha256(repository_path),
            },
            "attribution": {
                "path": attribution_path.name,
                "bytes": attribution_path.stat().st_size,
                "sha256": _sha256(attribution_path),
            },
        }
        report = {
            "format": REPORT_FORMAT,
            "format_version": FORMAT_VERSION,
            "status": (
                "bounded_profile_incomplete_not_training_authority"
                if missing
                else "bounded_profile_complete_not_training_authority"
            ),
            "plan_fingerprint": config["plan_fingerprint"],
            "source": {
                "repo": config["source"]["repo"],
                "revision": config["source"]["revision"],
                "release": config["source"]["release"],
                "files": source_identity,
            },
            "filters": config["filters"],
            "counts": dict(sorted(counts.items())),
            "accepted_languages": {
                language: {
                    "files": languages[language],
                    "bytes": language_bytes[language],
                    "sampled_bytes": sampled_bytes[language],
                    "target_bytes": targets[language],
                    "overshoot_bytes": sampled_bytes[language] - targets[language],
                }
                for language in targets
            },
            "sample_repositories": len(sample_repositories),
            "detected_licenses": dict(licenses.most_common()),
            "upstream_pii_placeholders": dict(pii),
            "high_confidence_secret_pattern_hits": dict(secrets),
            "outputs": outputs,
            "gates": {
                "source_revision_and_shards": "pass",
                "nested_schema": "pass",
                "released_content_size_and_sha256": "pass_for_sampled_files",
                "upstream_content_id_semantics": "preserved_and_plain_sha1_mismatches_reported",
                "restricted_filter_and_language_yield": "fail" if missing else "pass",
                "rights_and_attribution_acceptance": "pending_manual_review",
                "English_code_prose": "pending_comment_and_notebook_audit",
                "near_duplicates_cross_source": "pending",
                "benchmark_contamination": "pending",
                "secrets": "heuristic_counts_only_pending_dedicated_scanner",
                "training_authority": "blocked",
            },
        }
        if missing:
            report["missing_language_quotas"] = missing
        _write_json(staging / "report.json", report)
        if missing:
            raise RuntimeError(f"Stack v3 bounded shards did not fill language quotas: {missing}")
        os.replace(staging, output)
        if not config["output"]["keep_downloads"] and local_files is None:
            download_dir = Path(config["output"]["download_directory"])
            for item in config["source"]["files"]:
                (download_dir / item["path"]).unlink(missing_ok=True)
            shutil.rmtree(download_dir / ".cache", ignore_errors=True)
        return report
    except Exception:
        raise
