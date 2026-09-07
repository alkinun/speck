import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from speck.stack_v3_refine import _sample_partition
from speck.text_contamination import (
    scan_text_contamination,
    validate_text_contamination_config,
)

ROOT = Path(__file__).parents[1]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _record(text, host):
    return {
        "text": text,
        "released_content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "host": host,
        "url": f"https://{host}/page",
    }


def _config(tmp_path):
    exact = (
        "Electromagnetism thermodynamics crystallography astrophysics bioinformatics "
        "electrochemistry computational neuroscience biochemistry"
    )
    ngram = (
        "A careful silver telescope records seven unusual comets above the quiet northern horizon"
    )
    clean = (
        "Independent prose about gardens, weather, and careful observations in a local notebook."
    )
    records = [
        _record(exact, "exact.example"),
        _record(ngram, "ngram.example"),
        _record(clean, "clean.example"),
    ]
    source_path = tmp_path / "source.jsonl"
    source_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    source_hash = _sha256(source_path)
    parent = {
        "format": "parent_result",
        "status": "complete_not_training_authority",
        "sources": [
            {
                "id": "parent_source",
                "outputs": {"tokenizer_input": {"sha256": source_hash}},
            }
        ],
        "gates": {"training_authority": "blocked"},
    }
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(json.dumps(parent))
    benchmark_path = tmp_path / "benchmark.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"id": "exact", "question": exact},
                {"id": "ngram", "question": ngram + " before dawn during winter observations"},
            ]
        ),
        benchmark_path,
    )
    partition = {
        "seed": 42,
        "modulus": 10,
        "evaluation_remainders": [0],
        "training_bytes": 1,
        "evaluation_bytes": 1,
    }
    clean_split = _sample_partition(clean and records[-1], {**partition, "category": "web"})
    other_split = "eval" if clean_split == "train" else "train"
    filler = _record(f"Another clean {other_split} document " * 20, "filler.example")
    while _sample_partition(filler, {**partition, "category": "web"}) != other_split:
        filler["text"] += "x"
        filler["released_content_sha256"] = hashlib.sha256(filler["text"].encode()).hexdigest()
    with source_path.open("a") as handle:
        handle.write(json.dumps(filler) + "\n")
    source_hash = _sha256(source_path)
    parent["sources"][0]["outputs"]["tokenizer_input"]["sha256"] = source_hash
    parent_path.write_text(json.dumps(parent))
    return {
        "format": "speck_text_contamination",
        "format_version": 1,
        "status": "scan_authorized_not_training_authority",
        "sources": [
            {
                "id": "source",
                "category": "web",
                "path": str(source_path),
                "sha256": source_hash,
                "parent_report": str(parent_path),
                "parent_report_sha256": _sha256(parent_path),
                "parent_format": parent["format"],
                "parent_status": parent["status"],
                "parent_source_id": "parent_source",
                "partition": partition,
            }
        ],
        "benchmarks": [
            {
                "id": "fixture",
                "source": "example/benchmark",
                "revision": "a" * 40,
                "official_url": "https://example.com/benchmark",
                "split": "test",
                "path": str(benchmark_path),
                "sha256": _sha256(benchmark_path),
                "format": "parquet",
                "task_id_field": "id",
                "text_fields": ["question"],
                "expected_tasks": 2,
            }
        ],
        "policy": {
            "normalization": "NFKC+lower+unicode-word-punctuation",
            "token_pattern": "[^\\W_]+(?:['’][^\\W_]+)*|_+|[^\\s\\w]",
            "primary_ngram": 7,
            "sensitivity_ngram": 5,
            "minimum_alphanumeric_tokens": 3,
            "minimum_unique_tokens": 3,
            "maximum_tasks_per_ngram": 1,
            "critical_primary_matches": 2,
            "sensitivity_matches": 2,
            "minimum_exact_field_characters": 100,
            "minimum_exact_field_tokens": 6,
            "exact_anchor_tokens": 4,
        },
        "output_directory": str(tmp_path / "output"),
    }


def test_text_contamination_removes_independent_exact_and_ngram_matches(tmp_path):
    config = validate_text_contamination_config(_config(tmp_path))
    report = scan_text_contamination(config)
    source = report["sources"][0]

    assert source["counts"]["records_removed_critical"] == 2
    assert source["counts"]["records_retained"] == 2
    assert source["partition"]["train_bytes"] >= 1
    assert source["partition"]["eval_bytes"] >= 1
    assert source["critical_removed"][0]["exact_field_indices"] == {"fixture:exact": [0]}
    assert all("text" not in item for item in source["critical_removed"])
    assert report["gates"]["training_authority"] == "blocked"


def test_text_contamination_rejects_duplicate_task_references(tmp_path):
    raw = _config(tmp_path)
    benchmark = raw["benchmarks"][0]
    path = Path(benchmark["path"])
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"id": "duplicate", "question": "First sufficiently detailed benchmark question."},
                {"id": "duplicate", "question": "Second sufficiently detailed benchmark question."},
            ]
        ),
        path,
    )
    benchmark["sha256"] = _sha256(path)
    config = validate_text_contamination_config(raw)
    with pytest.raises(ValueError, match="duplicate benchmark task reference"):
        scan_text_contamination(config)


def test_flagship_web_contamination_plan_freezes_complete_short_context_inventory():
    path = ROOT / "research" / "flagship" / "web_contamination_v1.json"
    config = validate_text_contamination_config(
        json.loads(path.read_text()), config_dir=path.parent
    )

    assert [source["id"] for source in config["sources"]] == [
        "fineweb_edu",
        "ultrafineweb_en_v1_4",
        "dclm_baseline",
        "fineweb_base",
    ]
    assert len(config["benchmarks"]) == 20
    assert sum(benchmark["expected_tasks"] for benchmark in config["benchmarks"]) == 63_652
    assert {benchmark["id"] for benchmark in config["benchmarks"]} >= {
        "hellaswag",
        "arc-easy",
        "arc-challenge",
        "piqa",
        "winogrande-xl",
        "lambada-openai-en",
        "mmlu-all",
        "commonsenseqa",
        "triviaqa-rc-nocontext",
        "gsm8k-main",
        "math-algebra",
        "humaneval",
        "mbpp",
        "bigcodebench-v0.1.4",
    }
    assert config["policy"]["maximum_tasks_per_ngram"] == 1
    assert config["policy"]["minimum_exact_field_tokens"] < config["policy"]["primary_ngram"]
