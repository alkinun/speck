import hashlib
import json
from pathlib import Path

from speck.code_contamination import (
    scan_code_contamination,
    validate_contamination_config,
)
from speck.stack_v3_refine import _sample_partition

ROOT = Path(__file__).parents[1]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _record(text, identity, repo):
    return {
        "text": text,
        "content_id": identity * 40,
        "released_content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "repo_path": repo,
        "repo_id": 1,
        "commit_id": "a" * 40,
        "file_path": f"{identity}.py",
        "language": "Python",
        "detected_licenses": ["MIT"],
        "size_bytes": len(text.encode()),
    }


def _config(tmp_path):
    benchmark_code = (
        "def calculate_special_total(values):\n    return sum(value * value for value in values)\n"
    )
    contaminated = _record(
        benchmark_code + "print(calculate_special_total([1, 2]))\n", "1", "bad/repo"
    )
    clean = _record(
        "def unrelated_parser(payload):\n    return {'length': len(payload), 'empty': not payload}\n",
        "2",
        "good/repo",
    )
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(json.dumps(contaminated) + "\n" + json.dumps(clean) + "\n")
    input_hash = _sha256(input_path)
    parent_report = {
        "format": "speck_stack_v3_refinement_result",
        "status": "refinement_complete_not_training_authority",
        "outputs": {"tokenizer_input": {"sha256": input_hash}},
        "gates": {"training_authority": "blocked"},
    }
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(json.dumps(parent_report))
    benchmark_path = tmp_path / "benchmark.jsonl"
    benchmark_path.write_text(
        json.dumps(
            {
                "task_id": "task-1",
                "prompt": "Write calculate_special_total for a list of integer values.",
                "solution": benchmark_code,
                "tests": ["assert calculate_special_total([1, 2]) == 5"],
            }
        )
        + "\n"
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
        "format": "speck_code_contamination",
        "format_version": 1,
        "status": "scan_authorized_not_training_authority",
        "input": {
            "path": str(input_path),
            "sha256": input_hash,
            "report": str(parent_path),
            "report_sha256": _sha256(parent_path),
        },
        "benchmarks": [
            {
                "id": "test-benchmark",
                "source": "example/benchmark",
                "revision": "a" * 40,
                "official_url": "https://example.com/benchmark",
                "path": str(benchmark_path),
                "sha256": _sha256(benchmark_path),
                "format": "jsonl",
                "task_id_field": "task_id",
                "text_fields": ["prompt", "solution", "tests"],
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


def test_code_contamination_removes_critical_match_and_preserves_clean_quota(tmp_path):
    config = validate_contamination_config(_config(tmp_path))
    report = scan_code_contamination(config)
    output = Path(config["output_directory"])

    assert report["status"] == "decontamination_complete_not_training_authority"
    assert report["index"]["tasks"] == 1
    assert report["counts"]["records_seen"] == 2
    assert report["counts"]["records_removed_critical"] == 1
    assert report["counts"]["records_retained"] == 1
    assert report["critical_removed"][0]["critical_tasks"] == ["test-benchmark:task-1"]
    assert "text" not in report["critical_removed"][0]
    retained = json.loads((output / "tokenizer-input.jsonl").read_text())
    assert retained["repo_path"] == "good/repo"
    assert report["gates"]["training_authority"] == "blocked"


def test_real_code_contamination_plan_pins_three_benchmark_payloads():
    path = ROOT / "research" / "flagship" / "code_contamination_v1.json"
    config = validate_contamination_config(json.loads(path.read_text()), config_dir=path.parent)

    assert [benchmark["id"] for benchmark in config["benchmarks"]] == [
        "humaneval",
        "mbpp",
        "bigcodebench-v0.1.4",
    ]
    assert sum(benchmark["expected_tasks"] for benchmark in config["benchmarks"]) == 2_278
    assert config["policy"]["primary_ngram"] == 13
    assert config["policy"]["sensitivity_ngram"] == 10
    assert config["downstream_partition"]["training_bytes"] == 55_000_000


def test_recorded_code_contamination_binds_code_and_preserves_quota():
    result = json.loads(
        (ROOT / "results" / "data" / "stack-v3.1-code-contamination-20260907.json").read_text()
    )
    implementation = result["implementation"]
    scan = result["result"]

    assert result["status"] == "code_benchmark_decontamination_pass_training_authority_blocked"
    assert implementation["config_sha256"] == _sha256(ROOT / implementation["config"])
    assert implementation["module_sha256"] == _sha256(ROOT / implementation["module"])
    assert implementation["cli_sha256"] == _sha256(ROOT / implementation["cli"])
    assert sum(benchmark["tasks"] for benchmark in result["benchmarks"]) == 2_278
    assert scan["records_removed_critical"] == 125
    assert scan["exact_field_matches"] == 0
    assert scan["training_partition"]["bytes"] >= scan["training_partition"]["target_bytes"]
    assert scan["evaluation_partition"]["bytes"] >= scan["evaluation_partition"]["target_bytes"]
    assert "cross-source near-duplicate analysis" in result["blocked_gates"]
