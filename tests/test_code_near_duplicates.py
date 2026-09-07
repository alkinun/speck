import hashlib
import json
from pathlib import Path

import pytest

from speck.code_near_duplicates import (
    analyze_cross_source_duplicates,
    validate_duplicate_config,
)
from speck.stack_v3_refine import _sample_partition

ROOT = Path(__file__).parents[1]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _record(text, identity, repo):
    return {
        "text": text,
        "released_content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "content_id": identity * 40,
        "repo_path": repo,
        "commit_id": "a" * 40,
        "file_path": f"{identity}.py",
        "language": "Python",
        "detected_licenses": ["MIT"],
    }


def _source(tmp_path, source_id, precedence, records, partition, targets=True):
    path = tmp_path / f"{source_id}.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    digest = _sha256(path)
    report = {
        "format": f"{source_id}_report",
        "status": "complete_not_training_authority",
        "outputs": {"tokenizer_input": {"sha256": digest}},
        "gates": {"training_authority": "blocked"},
    }
    report_path = tmp_path / f"{source_id}-report.json"
    report_path.write_text(json.dumps(report))
    training = evaluation = 0
    if targets:
        chosen = _sample_partition(records[0], partition)
        if chosen == "train":
            training = 1
        else:
            evaluation = 1
    return {
        "id": source_id,
        "precedence": precedence,
        "path": str(path),
        "sha256": digest,
        "parent_report": str(report_path),
        "parent_report_sha256": _sha256(report_path),
        "parent_format": report["format"],
        "parent_status": report["status"],
        "training_bytes": training,
        "evaluation_bytes": evaluation,
    }


def _config(tmp_path):
    base = " ".join(f"token_{index}" for index in range(100))
    near = base.replace("token_50", "changed_token")
    clean = " ".join(f"independent_{index}" for index in range(100))
    partition = {
        "seed": 42,
        "category": "code",
        "modulus": 10,
        "evaluation_remainders": [0],
    }
    base_record = _record(base, "1", "first/repo")
    sources = [
        _source(tmp_path, "first", 1, [base_record], partition),
        _source(
            tmp_path,
            "second",
            2,
            [
                _record(base, "1", "second/exact"),
                _record(near, "2", "second/near"),
            ],
            partition,
            targets=False,
        ),
        _source(tmp_path, "third", 3, [_record(clean, "3", "third/clean")], partition),
    ]
    return {
        "format": "speck_code_cross_source_duplicates",
        "format_version": 1,
        "status": "analysis_authorized_not_training_authority",
        "sources": sources,
        "policy": {
            "normalization": "NFKC+lower+lexical-code-tokens",
            "token_pattern": "[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[^\\s]",
            "shingle_tokens": 3,
            "minimum_document_tokens": 5,
            "maximum_document_tokens": 1_000,
            "num_perm": 64,
            "minhash_seed": 42,
            "lsh_threshold": 0.5,
            "verified_jaccard_threshold": 0.5,
            "partition_seed": 42,
            "partition_modulus": 10,
            "partition_evaluation_remainders": [0],
        },
        "output_directory": str(tmp_path / "output"),
    }


def test_cross_source_precedence_removes_exact_and_verified_near_duplicates(tmp_path):
    pytest.importorskip("datasketch")
    config = validate_duplicate_config(_config(tmp_path))
    report = analyze_cross_source_duplicates(config)

    assert report["status"] == "cross_source_duplicates_complete_not_training_authority"
    assert len(report["exact_cross_source_matches"]) == 1
    assert len(report["near_cross_source_matches"]) == 1
    assert report["exact_cross_source_matches"][0]["kept_source"] == "first"
    assert report["near_cross_source_matches"][0]["kept_source"] == "first"
    assert report["near_cross_source_matches"][0]["verified_shingle_jaccard"] > 0.9
    assert report["gates"]["training_authority"] == "blocked"
    assert Path(config["output_directory"], "second.jsonl").read_text() == ""


def test_real_cross_source_plan_freezes_precedence_and_minhash_policy():
    path = ROOT / "research" / "flagship" / "code_cross_source_duplicates_v1.json"
    config = validate_duplicate_config(json.loads(path.read_text()), config_dir=path.parent)

    assert [source["id"] for source in config["sources"]] == [
        "stack_edu",
        "stack_v3",
        "common_pile_stackv2_edu",
    ]
    assert config["policy"]["num_perm"] == 128
    assert config["policy"]["lsh_threshold"] == 0.8
    assert config["policy"]["verified_jaccard_threshold"] == 0.8
    assert sum(source["training_bytes"] for source in config["sources"]) == 85_000_000


def test_recorded_overlap_binds_implementations_and_preserves_scope_limit():
    result = json.loads(
        (ROOT / "results" / "data" / "code-source-overlap-20260907.json").read_text()
    )
    implementation = result["implementation"]
    overlap = result["overlap"]

    assert result["status"] == "bounded_cross_source_overlap_pass_training_authority_blocked"
    for path_key, hash_key in (
        ("gitleaks_module", "gitleaks_module_sha256"),
        ("gitleaks_cli", "gitleaks_cli_sha256"),
        ("common_pile_config", "common_pile_config_sha256"),
        ("common_pile_module", "common_pile_module_sha256"),
        ("common_pile_cli", "common_pile_cli_sha256"),
        ("stack_edu_gitleaks_config", "stack_edu_gitleaks_config_sha256"),
        ("common_pile_gitleaks_config", "common_pile_gitleaks_config_sha256"),
        ("overlap_config", "overlap_config_sha256"),
        ("overlap_module", "overlap_module_sha256"),
        ("overlap_cli", "overlap_cli_sha256"),
    ):
        assert implementation[hash_key] == _sha256(ROOT / implementation[path_key])
    assert result["dependency"]["pyproject_sha256"] == _sha256(ROOT / "pyproject.toml")
    assert result["dependency"]["lock_sha256"] == _sha256(ROOT / "uv.lock")
    assert overlap["retained_documents"] == 31_583
    assert overlap["exact_released_content_matches"] == 0
    assert overlap["verified_near_duplicate_matches"] == 0
    assert "bounded" in overlap["scope_limit"]
    assert "Stack-Edu and Common Pile code benchmark contamination" in result["blocked_gates"]
