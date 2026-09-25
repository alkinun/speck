import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from speck.evaluation.protocol import BenchmarkExclusion, prepare_protocol
from speck.provenance.io import file_sha256


def fixture_protocol(tmp_path):
    policy = json.loads(
        (Path(__file__).parents[2] / "experiments/pilot/evaluation.json").read_text()
    )["exclusion_policy"]
    rows = [
        {
            "id": str(i),
            "question": f"A distinctive arithmetic exercise about planet {i} and its moons requires a precise calculation of the orbital period.",
        }
        for i in range(30)
    ]
    rows.append({**rows[0], "id": "duplicate-prompt"})
    source = tmp_path / "questions.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    entry = {
        "id": "fixture",
        "repo": "local/fixture",
        "revision": "a" * 40,
        "filename": source.name,
        "sha256": file_sha256(source),
        "format": "jsonl",
        "text_fields": ["question"],
        "task_id_field": "id",
        "expected_tasks": len(rows),
    }
    path = tmp_path / "protocol.json"
    path.write_text(
        json.dumps(
            {
                "format": "speck_capability_protocol",
                "format_version": 1,
                "development_fraction": 0.2,
                "seed": 42,
                "datasets": [entry],
                "exclusion_policy": policy,
            }
        )
    )
    return path, source, rows


def test_protocol_is_repeatable_disjoint_and_groups_duplicate_prompts(tmp_path):
    config, source, _ = fixture_protocol(tmp_path)

    def fetch(*args, **kwargs):
        return source

    output = tmp_path / "prepared.json"
    first = prepare_protocol(config, output, fetch=fetch)
    assert prepare_protocol(config, output, fetch=fetch) == first
    splits = first["partitions"]["fixture"]
    assert set(splits["development"]).isdisjoint(splits["final"])
    assert sum(map(len, splits.values())) == 31
    assert next(name for name, ids in splits.items() if "0" in ids) == next(
        name for name, ids in splits.items() if "duplicate-prompt" in ids
    )
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare_protocol(config, output, fetch=fetch)


def test_benchmark_exclusion_catches_embedded_questions_without_model_outputs(tmp_path):
    config, source, rows = fixture_protocol(tmp_path)
    result = prepare_protocol(config, tmp_path / "prepared.json", fetch=lambda *a, **k: source)
    exclusion = BenchmarkExclusion(result)
    assert exclusion.matches("A preamble. " + rows[5]["question"] + " Here is the answer.")
    assert not exclusion.matches("A completely unrelated description of cooking vegetables.")


POLICY = {
    "token_pattern": "[^\\W_]+(?:['’][^\\W_]+)*|_+|[^\\s\\w]",
    "primary_ngram": 7,
    "sensitivity_ngram": 5,
    "minimum_alphanumeric_tokens": 3,
    "minimum_unique_tokens": 3,
    "critical_primary_matches": 2,
    "sensitivity_matches": 2,
    "minimum_exact_field_characters": 100,
    "minimum_exact_field_tokens": 6,
    "exact_anchor_tokens": 4,
}
EXACT = (
    "Electromagnetism thermodynamics crystallography astrophysics bioinformatics "
    "electrochemistry computational neuroscience biochemistry"
)
NGRAM = "A careful silver telescope records seven unusual comets above the quiet northern horizon"


def parquet_protocol(tmp_path, rows):
    path = tmp_path / "benchmark.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    benchmark = {
        "id": "fixture",
        "path": str(path),
        "sha256": file_sha256(path),
        "format": "parquet",
        "task_id_field": "id",
        "text_fields": ["question"],
        "expected_tasks": len(rows),
    }
    return {"benchmarks": [benchmark], "policy": POLICY}


def test_benchmark_exclusion_matches_exact_fields_and_task_unique_ngrams(tmp_path):
    exclusion = BenchmarkExclusion(
        parquet_protocol(
            tmp_path,
            [
                {"id": "exact", "question": EXACT},
                {"id": "ngram", "question": NGRAM + " before dawn during winter observations"},
            ],
        )
    )
    assert exclusion.matches(f"Intro. {EXACT}. Outro.") == ["fixture:exact"]
    assert exclusion.matches(NGRAM) == ["fixture:ngram"]
    assert not exclusion.matches(
        "Independent prose about gardens, weather, and careful observations in a local notebook."
    )


def test_benchmark_exclusion_rejects_duplicate_task_references(tmp_path):
    protocol = parquet_protocol(
        tmp_path,
        [
            {"id": "duplicate", "question": "First sufficiently detailed benchmark question."},
            {"id": "duplicate", "question": "Second sufficiently detailed benchmark question."},
        ],
    )
    with pytest.raises(ValueError, match="duplicate benchmark task reference"):
        BenchmarkExclusion(protocol)
