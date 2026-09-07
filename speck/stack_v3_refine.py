"""Refine a bounded Stack v3 sample through license, secret, and English gates."""

import ast
import hashlib
import io
import json
import os
import re
import shutil
import tokenize
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path

FORMAT = "speck_stack_v3_refinement"
FORMAT_VERSION = 1
REPORT_FORMAT = "speck_stack_v3_refinement_result"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_C_STYLE_COMMENTS = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
_SQL_COMMENTS = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)


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


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _exact_keys(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{name} must contain exactly: {', '.join(sorted(keys))}")


def _path(value, name, config_dir):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(value).expanduser()
    return str((config_dir / path).resolve() if not path.is_absolute() else path.resolve())


def validate_refinement_config(config, *, config_dir=None):
    """Validate and normalize a Stack v3 refinement plan."""

    config_dir = Path(config_dir or ".").resolve()
    _exact_keys(
        config,
        {
            "format",
            "format_version",
            "status",
            "input",
            "scanner",
            "license_policy",
            "English_prose",
            "downstream_partition",
            "output_directory",
        },
        "Stack v3 refinement",
    )
    if config["format"] != FORMAT or config["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported Stack v3 refinement format")
    if config["status"] != "refinement_authorized_not_training_authority":
        raise ValueError("Stack v3 refinement must remain non-authoritative")

    source = config["input"]
    _exact_keys(
        source,
        {
            "tokenizer_jsonl",
            "tokenizer_jsonl_sha256",
            "repository_jsonl",
            "repository_jsonl_sha256",
            "attribution_jsonl",
            "attribution_jsonl_sha256",
        },
        "input",
    )
    normalized_input = {}
    for key in ("tokenizer_jsonl", "repository_jsonl", "attribution_jsonl"):
        normalized_input[key] = _path(source[key], f"input.{key}", config_dir)
        digest = source[f"{key}_sha256"]
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError(f"input.{key}_sha256 must be lowercase hexadecimal")
        normalized_input[f"{key}_sha256"] = digest

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
    if scanner["name"] != "gitleaks" or not isinstance(scanner["version"], str):
        raise ValueError("scanner must identify a pinned Gitleaks version")
    if not isinstance(scanner["official_url"], str) or not scanner["official_url"].startswith(
        "https://github.com/gitleaks/gitleaks"
    ):
        raise ValueError("scanner official URL must reference Gitleaks upstream")
    normalized_scanner = dict(scanner)
    for key in ("binary", "report"):
        normalized_scanner[key] = _path(scanner[key], f"scanner.{key}", config_dir)
    for key in ("binary_sha256", "release_archive_sha256", "report_sha256"):
        if not isinstance(scanner[key], str) or not _SHA256.fullmatch(scanner[key]):
            raise ValueError(f"scanner.{key} must be lowercase hexadecimal")
    if scanner["redaction_percent"] != 100:
        raise ValueError("Gitleaks reports must use 100 percent redaction")

    policy = config["license_policy"]
    _exact_keys(
        policy,
        {"name", "accepted_detected_licenses", "manual_legal_acceptance"},
        "license_policy",
    )
    if not isinstance(policy["name"], str) or not policy["name"]:
        raise ValueError("license_policy.name must be non-empty")
    licenses = policy["accepted_detected_licenses"]
    if (
        not isinstance(licenses, list)
        or not licenses
        or any(not isinstance(value, str) or not value for value in licenses)
        or len(licenses) != len(set(licenses))
    ):
        raise ValueError("accepted_detected_licenses must be a unique non-empty string list")
    if policy["manual_legal_acceptance"] != "pending":
        raise ValueError("engineering license filtering cannot assert legal acceptance")

    english = config["English_prose"]
    _exact_keys(
        english,
        {"detector", "languages", "minimum_alphabetic_characters", "minimum_probability"},
        "English_prose",
    )
    if english["detector"] != "py3langid==0.3.0":
        raise ValueError("English prose filtering must use pinned py3langid==0.3.0")
    languages = english["languages"]
    if (
        not isinstance(languages, list)
        or not languages
        or any(not isinstance(value, str) or not value for value in languages)
        or len(languages) != len(set(languages))
    ):
        raise ValueError("English prose languages must be unique non-empty strings")
    minimum_letters = _integer(
        english["minimum_alphabetic_characters"], "minimum_alphabetic_characters", 1
    )
    probability = english["minimum_probability"]
    if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not 0 < probability <= 1:
        raise ValueError("minimum_probability must be in (0, 1]")

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
    seed = _integer(partition["seed"], "downstream seed")
    if partition["category"] != "code":
        raise ValueError("Stack v3 downstream category must be code")
    modulus = _integer(partition["modulus"], "downstream modulus", 2)
    remainders = partition["evaluation_remainders"]
    if (
        not isinstance(remainders, list)
        or not remainders
        or len(remainders) != len(set(remainders))
        or any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < modulus for value in remainders)
        or len(remainders) == modulus
    ):
        raise ValueError("invalid downstream evaluation remainders")
    training_bytes = _integer(partition["training_bytes"], "downstream training bytes")
    evaluation_bytes = _integer(partition["evaluation_bytes"], "downstream evaluation bytes")
    if training_bytes + evaluation_bytes < 1:
        raise ValueError("downstream partition must request at least one byte")

    normalized = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "status": config["status"],
        "input": normalized_input,
        "scanner": normalized_scanner,
        "license_policy": {
            **policy,
            "accepted_detected_licenses": sorted(licenses),
        },
        "English_prose": {
            **english,
            "languages": list(languages),
            "minimum_alphabetic_characters": minimum_letters,
            "minimum_probability": float(probability),
        },
        "downstream_partition": {
            **partition,
            "seed": seed,
            "modulus": modulus,
            "evaluation_remainders": sorted(remainders),
            "training_bytes": training_bytes,
            "evaluation_bytes": evaluation_bytes,
        },
        "output_directory": _path(config["output_directory"], "output_directory", config_dir),
    }
    normalized["plan_fingerprint"] = _fingerprint(normalized)
    return normalized


def load_refinement_config(path):
    path = Path(path).resolve()
    return validate_refinement_config(
        json.loads(path.read_text(encoding="utf-8")), config_dir=path.parent
    )


def _validated_config(config):
    if "plan_fingerprint" not in config:
        return validate_refinement_config(config)
    payload = {key: value for key, value in config.items() if key != "plan_fingerprint"}
    if config["plan_fingerprint"] != _fingerprint(payload):
        raise ValueError("normalized Stack v3 refinement fingerprint mismatch")
    return config


def _verify_file(path, expected):
    path = Path(path)
    if not path.is_file() or _sha256(path) != expected:
        raise ValueError(f"Stack v3 refinement input identity mismatch: {path}")
    return path


def _python_prose(text):
    values = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type == tokenize.COMMENT:
                values.append(token.string.lstrip("#"))
    except (IndentationError, SyntaxError, tokenize.TokenError):
        pass
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return "\n".join(values)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            value = ast.get_docstring(node, clean=False)
            if value:
                values.append(value)
    return "\n".join(values)


def _extract_prose(text, language):
    if language == "Markdown":
        return text
    if language == "Python":
        return _python_prose(text)
    if language == "Shell":
        return "\n".join(
            line.lstrip()[1:]
            for line in text.splitlines()
            if line.lstrip().startswith("#") and not line.lstrip().startswith("#!")
        )
    if language == "SQL":
        return "\n".join(_SQL_COMMENTS.findall(text))
    return "\n".join(_C_STYLE_COMMENTS.findall(text))


@lru_cache(maxsize=1)
def _language_identifier():
    from py3langid.langid import MODEL_FILE, LanguageIdentifier

    return LanguageIdentifier.from_pickled_model(MODEL_FILE, norm_probs=True)


def _english_prose_result(text, language, settings):
    prose = _extract_prose(text, language)
    alphabetic = sum(character.isalpha() for character in prose)
    if alphabetic < settings["minimum_alphabetic_characters"]:
        return "insufficient_prose", None
    detected, probability = _language_identifier().classify(prose)
    probability = float(probability)
    if detected == "en" and probability >= settings["minimum_probability"]:
        return "English", probability
    return "non_English", probability


def _load_security_blocklist(config):
    scanner = config["scanner"]
    binary = _verify_file(scanner["binary"], scanner["binary_sha256"])
    report_path = _verify_file(scanner["report"], scanner["report_sha256"])
    input_path = Path(config["input"]["tokenizer_jsonl"])
    findings = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(findings, list):
        raise ValueError("Gitleaks report must be a JSON list")
    lines = set()
    rules = Counter()
    for finding in findings:
        if not isinstance(finding, dict):
            raise ValueError("Gitleaks findings must be objects")
        if finding.get("Secret") != "REDACTED" or finding.get("Line") not in {"", None}:
            raise ValueError("Gitleaks report is not fully redacted")
        if Path(finding.get("File", "")).resolve() != input_path.resolve():
            raise ValueError("Gitleaks report targets an unexpected file")
        line = _integer(finding.get("StartLine"), "Gitleaks StartLine", 1)
        lines.add(line)
        rule = finding.get("RuleID")
        if not isinstance(rule, str) or not rule:
            raise ValueError("Gitleaks finding has no rule ID")
        rules[rule] += 1

    blocked = set()
    with input_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line_number not in lines:
                continue
            record = json.loads(line)
            digest = record.get("released_content_sha256")
            if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                raise ValueError("Gitleaks-affected record lacks released content identity")
            blocked.add(digest)
    if len(blocked) != len(lines):
        raise ValueError("Gitleaks line mapping did not resolve every affected record")
    return blocked, {
        "name": scanner["name"],
        "version": scanner["version"],
        "official_url": scanner["official_url"],
        "binary": str(binary),
        "binary_sha256": scanner["binary_sha256"],
        "release_archive_sha256": scanner["release_archive_sha256"],
        "report": str(report_path),
        "report_sha256": scanner["report_sha256"],
        "redaction_percent": scanner["redaction_percent"],
        "findings": len(findings),
        "affected_records": len(blocked),
        "rule_counts": dict(sorted(rules.items())),
        "excluded_released_content_sha256": sorted(blocked),
    }


def _sample_partition(record, settings):
    text = record["text"]
    normalized = " ".join(unicodedata.normalize("NFKC", text).lower().split())
    digest = hashlib.sha256(normalized.encode()).hexdigest()
    payload = f"{settings['seed']}\0{settings['category']}\0{digest}".encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return (
        "eval"
        if value % settings["modulus"] in settings["evaluation_remainders"]
        else "train"
    )


def _dump_line(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def refine_stack_v3(config, *, restart=False):
    """Produce a non-authoritative v2 sample with scanner/license/language filtering."""

    config = _validated_config(config)
    for key in ("tokenizer_jsonl", "repository_jsonl", "attribution_jsonl"):
        _verify_file(config["input"][key], config["input"][f"{key}_sha256"])
    blocked, scanner_summary = _load_security_blocklist(config)
    accepted_licenses = set(config["license_policy"]["accepted_detected_licenses"])
    output = Path(config["output_directory"])
    staging = output.with_name(output.name + ".building")
    if output.exists():
        raise FileExistsError(f"Stack v3 refinement already exists: {output}")
    if staging.exists():
        if not restart:
            raise FileExistsError(f"incomplete Stack v3 refinement exists: {staging}")
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    tokenizer_path = staging / "tokenizer-input.jsonl"
    repository_path = staging / "repository-sample.jsonl"
    attribution_path = staging / "attribution.jsonl"
    counts = Counter()
    languages = Counter()
    language_bytes = Counter()
    licenses = Counter()
    partitions = Counter()
    probabilities = []
    repositories = set()
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
        Path(config["input"]["tokenizer_jsonl"]).open(encoding="utf-8") as source,
        tokenizer_path.open("w", encoding="utf-8") as tokenizer_output,
        repository_path.open("w", encoding="utf-8") as repository_output,
        attribution_path.open("w", encoding="utf-8") as attribution_output,
    ):
        for line_number, line in enumerate(source, 1):
            record = json.loads(line)
            counts["records_seen"] += 1
            text = record.get("text")
            digest = record.get("released_content_sha256")
            detected = record.get("detected_licenses")
            if (
                not isinstance(text, str)
                or not isinstance(digest, str)
                or hashlib.sha256(text.encode()).hexdigest() != digest
                or not isinstance(detected, list)
            ):
                raise ValueError(f"invalid Stack v3 v1 record at line {line_number}")
            if digest in blocked:
                counts["records_rejected_gitleaks"] += 1
                continue
            if not detected or any(license_id not in accepted_licenses for license_id in detected):
                counts["records_rejected_license_allowlist"] += 1
                continue
            language = record.get("language")
            if language not in config["English_prose"]["languages"]:
                raise ValueError(f"unexpected language in Stack v3 refinement: {language}")
            prose_result, probability = _english_prose_result(
                text, language, config["English_prose"]
            )
            counts[f"records_prose_{prose_result}"] += 1
            if probability is not None:
                probabilities.append(probability)
            if prose_result == "non_English":
                counts["records_rejected_non_English_prose"] += 1
                continue
            counts["records_accepted"] += 1
            size = len(text.encode())
            languages[language] += 1
            language_bytes[language] += size
            for license_id in detected:
                licenses[license_id] += 1
            partition = _sample_partition(record, config["downstream_partition"])
            partitions[f"{partition}_bytes"] += size
            partitions[f"{partition}_records"] += 1
            repo_path = record["repo_path"]
            if repo_path != current_repo:
                flush_repository(repository_output)
                current_repo = repo_path
                current_files = []
                current_metadata = {
                    "repo_id": record.get("repo_id"),
                    "commit_id": record.get("commit_id"),
                }
            repositories.add(repo_path)
            _dump_line(tokenizer_output, record)
            _dump_line(
                attribution_output,
                {key: value for key, value in record.items() if key != "text"},
            )
            current_files.append(record)
        flush_repository(repository_output)
        for handle in (tokenizer_output, repository_output, attribution_output):
            handle.flush()
            os.fsync(handle.fileno())

    partition_targets = {
        "train_bytes": config["downstream_partition"]["training_bytes"],
        "eval_bytes": config["downstream_partition"]["evaluation_bytes"],
    }
    missing = {
        split: {"bytes": partitions[split], "target_bytes": target}
        for split, target in partition_targets.items()
        if partitions[split] < target
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
            "refinement_incomplete_not_training_authority"
            if missing
            else "refinement_complete_not_training_authority"
        ),
        "plan_fingerprint": config["plan_fingerprint"],
        "input": config["input"],
        "scanner": scanner_summary,
        "license_policy": config["license_policy"],
        "English_prose": {
            **config["English_prose"],
            "classified_probability_min": min(probabilities) if probabilities else None,
            "classified_probability_max": max(probabilities) if probabilities else None,
        },
        "downstream_partition": {
            **config["downstream_partition"],
            "observed": dict(sorted(partitions.items())),
        },
        "counts": dict(sorted(counts.items())),
        "repositories": len(repositories),
        "accepted_languages": {
            language: {"records": languages[language], "bytes": language_bytes[language]}
            for language in config["English_prose"]["languages"]
        },
        "accepted_licenses": dict(licenses.most_common()),
        "outputs": outputs,
        "gates": {
            "input_artifact_identity": "pass",
            "gitleaks_binary_report_and_redaction": "pass",
            "gitleaks_affected_records_excluded": "pass",
            "engineering_license_allowlist": "pass",
            "manual_legal_acceptance": "pending",
            "English_comments_docs": "pass_for_declared_extractors_with_strings_out_of_scope",
            "downstream_tokenizer_partition_yield": "fail" if missing else "pass",
            "near_duplicates_cross_source": "pending",
            "benchmark_contamination": "pending",
            "acquisition_cleanup_and_resume": "pending",
            "training_authority": "blocked",
        },
    }
    if missing:
        report["missing_downstream_partition_bytes"] = missing
    _write_json(staging / "report.json", report)
    if missing:
        raise RuntimeError(f"Stack v3 refinement did not fill downstream partition: {missing}")
    os.replace(staging, output)
    return report
