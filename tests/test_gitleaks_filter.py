import hashlib
import json
from pathlib import Path

from speck.gitleaks_filter import (
    apply_gitleaks_filter,
    validate_gitleaks_filter_config,
)
from speck.stack_v3_refine import _sample_partition

ROOT = Path(__file__).parents[1]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _record(text, identity):
    return {
        "text": text,
        "released_content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "content_id": identity * 40,
        "repo_path": f"example/{identity}",
        "commit_id": None,
        "file_path": f"{identity}.rs",
        "language": "Rust",
        "detected_licenses": ["MIT"],
    }


def _config(tmp_path):
    blocked = _record("let leaked_value = 1;", "1")
    retained = _record("fn retained_value() -> usize { 2 }", "2")
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps(blocked) + "\n" + json.dumps(retained) + "\n")
    parent = {
        "format": "test_parent",
        "status": "complete_not_training_authority",
        "outputs": {"tokenizer_input": {"sha256": _sha256(source)}},
        "gates": {"training_authority": "blocked"},
    }
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(json.dumps(parent))
    binary = tmp_path / "gitleaks"
    binary.write_bytes(b"gitleaks-test")
    scanner = [
        {
            "RuleID": "generic-api-key",
            "Secret": "REDACTED",
            "Line": "",
            "File": str(source),
            "StartLine": 1,
        }
    ]
    scanner_path = tmp_path / "scanner.json"
    scanner_path.write_text(json.dumps(scanner))
    partition = {
        "seed": 42,
        "category": "code",
        "modulus": 10,
        "evaluation_remainders": [0],
        "training_bytes": 0,
        "evaluation_bytes": 0,
    }
    chosen = _sample_partition(retained, partition)
    partition["training_bytes" if chosen == "train" else "evaluation_bytes"] = 1
    return {
        "format": "speck_gitleaks_filter",
        "format_version": 1,
        "status": "exclusion_authorized_not_training_authority",
        "input": {
            "path": str(source),
            "sha256": _sha256(source),
            "parent_report": str(parent_path),
            "parent_report_sha256": _sha256(parent_path),
            "parent_format": "test_parent",
            "parent_status": "complete_not_training_authority",
        },
        "scanner": {
            "name": "gitleaks",
            "version": "test",
            "official_url": "https://github.com/gitleaks/gitleaks/releases/tag/test",
            "binary": str(binary),
            "binary_sha256": _sha256(binary),
            "release_archive_sha256": "a" * 64,
            "report": str(scanner_path),
            "report_sha256": _sha256(scanner_path),
            "redaction_percent": 100,
        },
        "downstream_partition": partition,
        "output_directory": str(tmp_path / "output"),
    }


def test_generic_gitleaks_filter_excludes_affected_record_and_preserves_quota(tmp_path):
    config = validate_gitleaks_filter_config(_config(tmp_path))
    report = apply_gitleaks_filter(config)
    output = Path(config["output_directory"])

    assert report["status"] == "gitleaks_exclusion_complete_not_training_authority"
    assert report["scanner"]["findings"] == report["scanner"]["affected_records"] == 1
    assert report["counts"]["records_rejected_gitleaks"] == 1
    assert report["counts"]["records_retained"] == 1
    assert report["gates"]["training_authority"] == "blocked"
    record = json.loads((output / "tokenizer-input.jsonl").read_text())
    attribution = json.loads((output / "attribution.jsonl").read_text())
    assert record["repo_path"] == "example/2"
    assert "text" not in attribution


def test_real_stack_edu_gitleaks_plan_is_fully_redacted_and_bounded():
    path = ROOT / "research" / "flagship" / "stack_edu_gitleaks_v1.json"
    config = validate_gitleaks_filter_config(json.loads(path.read_text()), config_dir=path.parent)

    assert config["scanner"]["version"] == "8.30.1"
    assert config["scanner"]["redaction_percent"] == 100
    assert config["downstream_partition"]["training_bytes"] == 15_000_000
    assert config["downstream_partition"]["evaluation_bytes"] == 1_500_000

    common_pile = ROOT / "research" / "flagship" / "common_pile_code_gitleaks_v1.json"
    common_pile_config = validate_gitleaks_filter_config(
        json.loads(common_pile.read_text()), config_dir=common_pile.parent
    )
    assert common_pile_config["scanner"]["report_sha256"] != config["scanner"]["report_sha256"]
    assert common_pile_config["downstream_partition"]["training_bytes"] == 10_000_000
    assert common_pile_config["downstream_partition"]["evaluation_bytes"] == 1_000_000
