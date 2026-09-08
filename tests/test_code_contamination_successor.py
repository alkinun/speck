import hashlib
import json
from pathlib import Path

import pytest

from speck.code_contamination_successor import (
    scan_code_contamination_successor,
    validate_successor_config,
)
from speck.stack_v3_refine import _sample_partition

ROOT = Path(__file__).parents[1]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _record(text, identity):
    return {
        "text": text,
        "released_content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "repo_path": f"repo/{identity}",
        "repo_id": identity,
        "commit_id": identity * 40,
        "file_path": f"{identity}.py",
        "language": "Python",
    }


def _config(tmp_path):
    benchmark_code = (
        "def special_total(values):\n    return sum(value * value for value in values)\n"
    )
    contaminated = _record(benchmark_code + "print(special_total([1, 2]))\n", "1")
    clean = _record("def unrelated(value):\n    return value is not None\n", "2")
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(json.dumps(contaminated) + "\n" + json.dumps(clean) + "\n")
    input_hash = _sha256(input_path)
    parent = {
        "format": "example_parent_result",
        "status": "complete_not_training_authority",
        "outputs": {"tokenizer_input": {"sha256": input_hash}},
        "gates": {"training_authority": "blocked"},
    }
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(json.dumps(parent))
    benchmark_path = tmp_path / "benchmark.jsonl"
    benchmark_path.write_text(
        json.dumps({"task_id": "one", "code": benchmark_code, "tests": []}) + "\n"
    )
    partition = {
        "seed": 42,
        "category": "code",
        "modulus": 10,
        "evaluation_remainders": [0],
        "training_bytes": 0,
        "evaluation_bytes": 0,
    }
    chosen = _sample_partition(clean, partition)
    partition["training_bytes" if chosen == "train" else "evaluation_bytes"] = 1
    return {
        "format": "speck_code_contamination_successor",
        "format_version": 1,
        "status": "scan_authorized_not_training_authority",
        "input": {
            "path": str(input_path),
            "sha256": input_hash,
            "report": str(parent_path),
            "report_sha256": _sha256(parent_path),
            "parent_format": parent["format"],
            "parent_status": parent["status"],
        },
        "benchmarks": [
            {
                "id": "example",
                "source": "example/source",
                "revision": "a" * 40,
                "official_url": "https://example.com",
                "path": str(benchmark_path),
                "sha256": _sha256(benchmark_path),
                "format": "jsonl",
                "task_id_field": "task_id",
                "text_fields": ["code", "tests"],
                "expected_tasks": 1,
            }
        ],
        "policy": {
            "normalization": "NFKC+lower+lexical-code-tokens",
            "token_pattern": "[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[^\\s]",
            "primary_ngram": 3,
            "sensitivity_ngram": 2,
            "minimum_alphanumeric_tokens": 2,
            "minimum_unique_tokens": 2,
            "maximum_tasks_per_ngram": 1,
            "critical_primary_matches": 2,
            "sensitivity_matches": 2,
            "minimum_exact_field_characters": 20,
        },
        "downstream_partition": partition,
        "output_directory": str(tmp_path / "output"),
    }


def test_successor_accepts_generic_parent_and_removes_critical_match(tmp_path):
    config = validate_successor_config(_config(tmp_path))
    report = scan_code_contamination_successor(config)

    assert report["status"] == "decontamination_complete_not_training_authority"
    assert report["counts"]["records_removed_critical"] == 1
    assert report["counts"]["records_retained"] == 1
    assert report["gates"]["training_authority"] == "blocked"


def test_successor_rejects_wrong_parent_identity(tmp_path):
    raw = _config(tmp_path)
    raw["input"]["parent_status"] = "wrong"
    with pytest.raises(ValueError, match="parent report"):
        scan_code_contamination_successor(validate_successor_config(raw))


def test_real_successor_plans_reuse_identical_benchmark_policy():
    original = json.loads((ROOT / "research/flagship/code_contamination_v1.json").read_text())
    for name, train, evaluation in (
        ("stack_edu_code_contamination_v1.json", 20_000_000, 2_000_000),
        ("common_pile_code_contamination_v1.json", 10_000_000, 1_000_000),
    ):
        path = ROOT / "research/flagship" / name
        config = validate_successor_config(json.loads(path.read_text()), config_dir=path.parent)
        assert config["benchmarks"] == original["benchmarks"]
        assert config["policy"] == original["policy"]
        assert config["downstream_partition"]["training_bytes"] == train
        assert config["downstream_partition"]["evaluation_bytes"] == evaluation


def test_recorded_successors_bind_implementation_and_preserve_quotas():
    result = json.loads(
        (ROOT / "results/data/code-contamination-successors-20260907.json").read_text()
    )
    implementation = result["implementation"]

    assert result["status"] == (
        "bounded_code_benchmark_decontamination_pass_training_authority_blocked"
    )
    for key in (
        "module",
        "frozen_base_module",
        "partition_module",
        "cli",
        "stack_edu_config",
        "common_pile_config",
    ):
        assert implementation[f"{key}_sha256"] == _sha256(ROOT / implementation[key])
    assert result["frozen_benchmark_policy"]["tasks"] == 2_278
    assert result["stack_edu"]["records_removed_critical"] == 32
    assert result["common_pile_stackv2_edu"]["records_removed_critical"] == 14
    for source in (result["stack_edu"], result["common_pile_stackv2_edu"]):
        assert source["training_partition_bytes"] >= source["training_target_bytes"]
        assert source["evaluation_partition_bytes"] >= source["evaluation_target_bytes"]
        assert source["records_with_exact_field_match"] == 0
    prior = result["prior_bounded_cross_source_analysis"]
    assert prior["result_sha256"] == _sha256(ROOT / prior["result"])
    assert prior["exact_released_content_matches"] == 0
    assert prior["verified_near_duplicate_matches"] == 0
