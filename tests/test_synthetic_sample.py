import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from speck.stack_v3_refine import _sample_partition
from speck.synthetic_sample import sample_synthetic_source, validate_synthetic_sample_config
from speck.text_contamination import validate_text_contamination_config
from speck.text_near_duplicates import validate_text_duplicate_config

ROOT = Path(__file__).parents[1]


def _text_for(split):
    settings = {"seed": 42, "category": "synthetic", "modulus": 2, "evaluation_remainders": [0]}
    for index in range(100):
        text = f"Synthetic English lesson {index}. This detailed explanation uses varied examples and clear reasoning for students."
        if _sample_partition({"text": text}, settings) == split:
            return text
    raise AssertionError("partition fixture failed")


def test_synthetic_sample_preserves_lineage_and_rejects_model_identity(tmp_path):
    rows = [
        {
            "id": "train",
            "text": _text_for("train"),
            "prompt": "write a lesson",
            "seed": "seed train",
            "style": "textbook",
        },
        {
            "id": "eval",
            "text": _text_for("eval"),
            "prompt": "write an article",
            "seed": "seed eval",
            "style": "blog",
        },
        {
            "id": "bad",
            "text": "As an AI language model, I cannot provide this requested educational article.",
            "prompt": "bad",
            "seed": "bad",
            "style": "refusal",
        },
    ]
    path = tmp_path / "synthetic.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    raw = {
        "format": "speck_synthetic_sample",
        "format_version": 1,
        "status": "bounded_sample_authorized_not_training_authority",
        "source": {
            "id": "example_synthetic",
            "repo": "example/synthetic",
            "revision": "a" * 40,
            "official_url": "https://example.com",
            "inputs": [
                {
                    "id": "generator_a",
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "rows": 3,
                    "format": "parquet",
                    "training_bytes": 1,
                    "evaluation_bytes": 1,
                    "generator": {
                        "id": "example/model",
                        "revision": "b" * 40,
                        "official_url": "https://example.com/model",
                        "license": "test",
                    },
                    "transformation": "lesson_generation",
                    "seed_source": {
                        "id": "example/seeds",
                        "revision": "c" * 40,
                        "official_url": "https://example.com/seeds",
                        "rights": "test",
                    },
                    "maximum_seed_text_jaccard": 0.8,
                    "fields": {
                        "text": "text",
                        "document_id": "id",
                        "prompt": "prompt",
                        "seed": "seed",
                        "style": "style",
                        "answer": None,
                        "url": None,
                        "domain": None,
                    },
                }
            ],
        },
        "rights": {
            "dataset_license": "test",
            "upstream_terms": "test",
            "authority": "manual_review_required",
            "redistribution": "undecided",
        },
        "filters": {
            "language_detector": "py3langid==0.3.0",
            "minimum_detected_English_probability": 0.8,
            "min_document_bytes": 20,
            "max_document_bytes": 10000,
            "maximum_duplicate_line_ratio": 0.5,
            "repetition_ngram_tokens": 5,
            "maximum_repeated_ngram_ratio": 0.5,
            "template_prefix_tokens": 5,
            "maximum_bytes_per_template_prefix": 10000,
            "model_identity_phrases": ["as an ai language model", "i cannot fulfill"],
            "allowed_email_placeholders": [],
            "allowed_ipv4_placeholders": [],
            "seed_overlap_shingle_tokens": 5,
            "maximum_bytes_per_seed_domain": 10000,
        },
        "downstream_partition": {
            "seed": 42,
            "category": "synthetic",
            "modulus": 2,
            "evaluation_remainders": [0],
        },
        "output_directory": str(tmp_path / "output"),
    }
    report = sample_synthetic_source(validate_synthetic_sample_config(raw))
    records = [
        json.loads(line)
        for line in (tmp_path / "output/tokenizer-input.jsonl").read_text().splitlines()
    ]

    assert len(records) == 2
    assert all(record["prompt_sha256"] and record["seed_sha256"] for record in records)
    assert all("prompt" not in record and "seed" not in record for record in records)
    assert report["counts"]["model_identity_phrase_rejected"] == 1
    assert report["style_profile"]["unique_styles"] == 2
    assert report["gates"]["training_authority"] == "blocked"


def test_flagship_synthetic_plans_freeze_sources_generators_and_margins():
    plans = []
    for path in sorted((ROOT / "research/flagship").glob("synthetic_*_v1.json")):
        raw = json.loads(path.read_text())
        if raw.get("format") == "speck_synthetic_sample":
            plans.append(validate_synthetic_sample_config(raw, config_dir=path.parent))

    assert {plan["source"]["id"] for plan in plans} == {
        "cosmopedia_v2",
        "ultrafineweb_l3_multi_style",
        "ultrafineweb_l3_qa",
        "megamath_qa",
    }
    inputs = [item for plan in plans for item in plan["source"]["inputs"]]
    assert len(inputs) == 5
    assert sum(item["training_bytes"] for item in inputs) == 120_000_000
    assert sum(item["evaluation_bytes"] for item in inputs) == 12_000_000
    assert all(item["generator"]["revision"] == "not_disclosed_by_dataset_card" for item in inputs)
    assert all(
        item["seed_source"]["revision"] == "not_disclosed_by_dataset_card" for item in inputs
    )


def test_flagship_synthetic_overlap_plan_freezes_precedence_and_quotas():
    path = ROOT / "research/flagship/synthetic_cross_source_duplicates_v1.json"
    plan = validate_text_duplicate_config(json.loads(path.read_text()), config_dir=path.parent)

    assert [source["id"] for source in plan["sources"]] == [
        "cosmopedia_v2",
        "ultrafineweb_l3_multi_style",
        "ultrafineweb_l3_qa",
        "megamath_qa",
    ]
    assert sum(source["training_bytes"] for source in plan["sources"]) == 100_000_000
    assert sum(source["evaluation_bytes"] for source in plan["sources"]) == 10_000_000
    assert plan["policy"]["category"] == "synthetic"


def test_flagship_synthetic_contamination_reuses_frozen_payload_policy():
    path = ROOT / "research/flagship/synthetic_contamination_v1.json"
    plan = validate_text_contamination_config(json.loads(path.read_text()), config_dir=path.parent)
    web_path = ROOT / "research/flagship/web_contamination_v1.json"
    web = validate_text_contamination_config(
        json.loads(web_path.read_text()), config_dir=web_path.parent
    )

    assert [source["id"] for source in plan["sources"]] == [
        "cosmopedia_v2",
        "ultrafineweb_l3_multi_style",
        "ultrafineweb_l3_qa",
        "megamath_qa",
    ]
    assert plan["policy"] == web["policy"]
    assert plan["benchmarks"] == web["benchmarks"]
