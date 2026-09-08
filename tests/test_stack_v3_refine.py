import hashlib
import json
from pathlib import Path

import pytest

from speck.stack_v3_refine import (
    _sample_partition,
    refine_stack_v3,
    validate_refinement_config,
)

ROOT = Path(__file__).parents[1]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_jsonl(path, values):
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


def _record(text, *, content_id, language="Python", licenses=None, repo="example/repo"):
    return {
        "text": text,
        "content_id": content_id,
        "released_content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "repo_path": repo,
        "repo_id": 1,
        "commit_id": "a" * 40,
        "file_path": f"src/{content_id}.py" if language == "Python" else "README.md",
        "language": language,
        "detected_licenses": licenses or ["MIT"],
        "size_bytes": len(text.encode()),
    }


def _config(tmp_path, *, unredacted=False):
    english = (
        '"""This module documents a deterministic English implementation with clear examples and '
        'careful behavior for every caller."""\nprint("ok")\n'
    )
    secret = "# sourcegraph access token belongs in the dedicated scanner test\nprint('blocked')\n"
    disallowed = "# This English file has a source license outside the conservative allowlist.\n"
    russian = (
        "Это подробная русская документация, которая должна быть отклонена английским языковым "
        "фильтром перед обучением токенизатора. " * 3
    )
    records = [
        _record(english, content_id="1" * 40),
        _record(secret, content_id="2" * 40),
        _record(disallowed, content_id="3" * 40, licenses=["CC0-1.0"]),
        _record(russian, content_id="4" * 40, language="Markdown"),
    ]
    tokenizer_path = tmp_path / "tokenizer.jsonl"
    repository_path = tmp_path / "repositories.jsonl"
    attribution_path = tmp_path / "attribution.jsonl"
    _write_jsonl(tokenizer_path, records)
    _write_jsonl(repository_path, [{"repo_path": "example/repo", "files": records}])
    _write_jsonl(
        attribution_path,
        [{key: value for key, value in record.items() if key != "text"} for record in records],
    )
    binary = tmp_path / "gitleaks"
    binary.write_bytes(b"pinned-gitleaks-binary")
    report_path = tmp_path / "gitleaks.json"
    report = [
        {
            "RuleID": "sourcegraph-access-token",
            "Secret": "not-redacted" if unredacted else "REDACTED",
            "Line": "" if not unredacted else "secret material",
            "File": str(tokenizer_path),
            "StartLine": 2,
        }
    ]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    partition = {
        "seed": 42,
        "category": "code",
        "modulus": 10,
        "evaluation_remainders": [0],
        "training_bytes": 0,
        "evaluation_bytes": 0,
    }
    chosen = _sample_partition(records[0], partition)
    if chosen == "train":
        partition["training_bytes"] = 1
    else:
        partition["evaluation_bytes"] = 1
    return {
        "format": "speck_stack_v3_refinement",
        "format_version": 1,
        "status": "refinement_authorized_not_training_authority",
        "input": {
            "tokenizer_jsonl": str(tokenizer_path),
            "tokenizer_jsonl_sha256": _sha256(tokenizer_path),
            "repository_jsonl": str(repository_path),
            "repository_jsonl_sha256": _sha256(repository_path),
            "attribution_jsonl": str(attribution_path),
            "attribution_jsonl_sha256": _sha256(attribution_path),
        },
        "scanner": {
            "name": "gitleaks",
            "version": "test",
            "official_url": "https://github.com/gitleaks/gitleaks/releases/tag/test",
            "binary": str(binary),
            "binary_sha256": _sha256(binary),
            "release_archive_sha256": "b" * 64,
            "report": str(report_path),
            "report_sha256": _sha256(report_path),
            "redaction_percent": 100,
        },
        "license_policy": {
            "name": "test",
            "accepted_detected_licenses": ["MIT"],
            "manual_legal_acceptance": "pending",
        },
        "English_prose": {
            "detector": "py3langid==0.3.0",
            "languages": ["Python", "Markdown"],
            "minimum_alphabetic_characters": 20,
            "minimum_probability": 0.6,
        },
        "downstream_partition": partition,
        "output_directory": str(tmp_path / "refined"),
    }


def test_refinement_excludes_scanner_license_and_non_english_records(tmp_path):
    config = validate_refinement_config(_config(tmp_path))
    report = refine_stack_v3(config)
    output = Path(config["output_directory"])

    assert report["status"] == "refinement_complete_not_training_authority"
    assert report["scanner"]["findings"] == 1
    assert report["scanner"]["affected_records"] == 1
    assert report["counts"]["records_seen"] == 4
    assert report["counts"]["records_accepted"] == 1
    assert report["counts"]["records_rejected_gitleaks"] == 1
    assert report["counts"]["records_rejected_license_allowlist"] == 1
    assert report["counts"]["records_rejected_non_English_prose"] == 1
    assert report["gates"]["manual_legal_acceptance"] == "pending"
    assert report["gates"]["training_authority"] == "blocked"

    record = json.loads((output / "tokenizer-input.jsonl").read_text())
    attribution = json.loads((output / "attribution.jsonl").read_text())
    assert record["released_content_sha256"] == attribution["released_content_sha256"]
    assert "text" not in attribution


def test_refinement_rejects_unredacted_scanner_reports(tmp_path):
    config = validate_refinement_config(_config(tmp_path, unredacted=True))
    with pytest.raises(ValueError, match="not fully redacted"):
        refine_stack_v3(config)


def test_real_refinement_plan_is_pinned_and_keeps_legal_gate_open():
    path = ROOT / "research" / "flagship" / "stack_v3_refinement_v2.json"
    config = validate_refinement_config(json.loads(path.read_text()), config_dir=path.parent)

    assert config["scanner"]["version"] == "8.30.1"
    assert config["scanner"]["redaction_percent"] == 100
    assert config["license_policy"]["manual_legal_acceptance"] == "pending"
    assert config["downstream_partition"]["training_bytes"] == 55_000_000
    assert config["downstream_partition"]["evaluation_bytes"] == 5_500_000

    successor = ROOT / "research" / "flagship" / "stack_v3_refinement_v2b.json"
    successor_config = validate_refinement_config(
        json.loads(successor.read_text()), config_dir=successor.parent
    )
    assert (
        successor_config["input"]["tokenizer_jsonl_sha256"]
        != config["input"]["tokenizer_jsonl_sha256"]
    )
    assert successor_config["scanner"]["report_sha256"] != config["scanner"]["report_sha256"]
    assert successor_config["license_policy"] == config["license_policy"]
    assert successor_config["English_prose"] == config["English_prose"]


def test_recorded_refinement_binds_code_and_keeps_training_blocked():
    result = json.loads(
        (ROOT / "results" / "data" / "stack-v3.1-refinement-20260907.json").read_text()
    )
    implementation = result["implementation"]
    passing = result["passing_refinement"]

    assert (
        result["status"]
        == "security_language_license_and_partition_pass_training_authority_blocked"
    )
    for path_key, hash_key in (
        ("expansion_config", "expansion_config_sha256"),
        ("expansion_module", "expansion_module_sha256"),
        ("expansion_cli", "expansion_cli_sha256"),
        ("failed_refinement_config", "failed_refinement_config_sha256"),
        ("passing_refinement_config", "passing_refinement_config_sha256"),
        ("refinement_module", "refinement_module_sha256"),
        ("refinement_cli", "refinement_cli_sha256"),
    ):
        assert implementation[hash_key] == _sha256(ROOT / implementation[path_key])
    assert result["scanner"]["findings"] == 37
    assert result["scanner"]["affected_records"] == 19
    assert passing["training_partition"]["bytes"] >= passing["training_partition"]["target_bytes"]
    assert (
        passing["evaluation_partition"]["bytes"] >= passing["evaluation_partition"]["target_bytes"]
    )
    assert result["engineering_license_policy"]["manual_legal_acceptance"] == "pending"
    assert "benchmark contamination analysis" in result["blocked_gates"]
