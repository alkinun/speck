import gzip
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from speck.science_sample import (
    _SPACED_OCR,
    _compressed_rows,
    _field,
    sample_science_source,
    validate_science_sample_config,
)
from speck.stack_v3_refine import _sample_partition
from speck.text_contamination import validate_text_contamination_config
from speck.text_near_duplicates import validate_text_duplicate_config

ROOT = Path(__file__).parents[1]


def _text_for(split):
    settings = {"seed": 42, "category": "science", "modulus": 2, "evaluation_remainders": [0]}
    for index in range(100):
        text = f"Scientific experiment {index} measures cellular protein activity. The study reports reproducible evidence, methods, and quantitative analysis."
        if _sample_partition({"text": text}, settings) == split:
            return text
    raise AssertionError("partition fixture failed")


def test_science_sample_preserves_text_and_rejects_ocr_and_license(tmp_path):
    rows = [
        {"id": "train", "text": _text_for("train"), "meta": {"license": "CC BY 4.0"}},
        {"id": "eval", "text": _text_for("eval"), "meta": {"license": "CC0"}},
        {
            "id": "ocr",
            "text": "A b s t r a c t scientific experiment protein analysis evidence methods.",
            "meta": {"license": "CC0"},
        },
        {
            "id": "blocked",
            "text": "Scientific experiment with protein and cellular analysis methods.",
            "meta": {"license": "unknown"},
        },
    ]
    path = tmp_path / "science.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    raw = {
        "format": "speck_science_sample",
        "format_version": 1,
        "status": "bounded_sample_authorized_not_training_authority",
        "source": {
            "id": "example_science",
            "repo": "example/science",
            "revision": "a" * 40,
            "official_url": "https://example.com",
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": 4,
            "file_format": "parquet",
            "fields": {
                "text": "text",
                "document_id": "id",
                "title": None,
                "url": None,
                "license": "meta.license",
                "language": None,
                "language_score": None,
                "secondary_language": None,
                "secondary_language_score": None,
                "quality_score": None,
                "date": None,
                "source_partition": None,
                "is_truncated": None,
                "extractor": None,
                "minhash_cluster_size": None,
                "duplicate_count": None,
            },
        },
        "rights": {
            "dataset_license": "test",
            "upstream_terms": "test",
            "authority": "manual_review_required",
            "redistribution": "undecided",
        },
        "filters": {
            "required_language": None,
            "minimum_language_score": None,
            "required_secondary_language": None,
            "minimum_secondary_language_score": None,
            "minimum_quality_score": None,
            "language_detector": "py3langid==0.3.0",
            "minimum_detected_English_probability": 0.8,
            "min_document_bytes": 20,
            "max_document_bytes": 10000,
            "minimum_alphabetic_ratio": 0.2,
            "maximum_duplicate_line_ratio": 0.5,
            "maximum_spaced_OCR_sequences": 0,
            "maximum_replacement_character_ratio": 0,
            "boilerplate_line_terms": ["cookie policy", "download pdf"],
            "maximum_boilerplate_line_ratio": 0.5,
            "accepted_document_licenses": ["CC BY 4.0", "CC0"],
            "minimum_science_term_hits": 2,
            "science_terms": ["analysis", "cellular", "experiment", "protein", "scientific"],
            "maximum_bytes_per_host": None,
            "allowed_email_placeholders": [],
            "allowed_ipv4_placeholders": [],
            "reject_truncated": False,
            "accepted_extractors": [],
            "maximum_minhash_cluster_size": None,
            "maximum_duplicate_count": None,
        },
        "downstream_partition": {
            "seed": 42,
            "category": "science",
            "modulus": 2,
            "evaluation_remainders": [0],
            "training_bytes": 1,
            "evaluation_bytes": 1,
        },
        "output_directory": str(tmp_path / "output"),
    }
    report = sample_science_source(validate_science_sample_config(raw))
    records = [
        json.loads(line)
        for line in (tmp_path / "output/tokenizer-input.jsonl").read_text().splitlines()
    ]

    assert {record["text"] for record in records} == {_text_for("train"), _text_for("eval")}
    assert _SPACED_OCR.search(rows[2]["text"])
    assert _field(rows[3], "meta.license") == "unknown"
    assert report["gates"]["training_authority"] == "blocked"


def test_science_sample_streams_gzip_and_zstd_jsonl(tmp_path):
    rows = [
        {"id": "one", "text": "scientific experiment"},
        {"id": "two", "text": "protein analysis"},
    ]
    payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
    gzip_path = tmp_path / "science.json.gz"
    with gzip.open(gzip_path, "wb") as handle:
        handle.write(payload)
    zstd_path = tmp_path / "science.jsonl.zst"
    with pa.output_stream(zstd_path, compression="zstd") as handle:
        handle.write(payload)

    assert list(_compressed_rows(gzip_path, "gzip")) == rows
    assert list(_compressed_rows(zstd_path, "zstd")) == rows


def test_flagship_science_plans_freeze_five_source_margins():
    plans = []
    for path in sorted((ROOT / "research/flagship").glob("science_*_v1.json")):
        raw = json.loads(path.read_text())
        if raw.get("format") == "speck_science_sample":
            plans.append(validate_science_sample_config(raw, config_dir=path.parent))

    assert {plan["source"]["id"] for plan in plans} == {
        "pes2o_v3",
        "finepdfs_edu_en",
        "common_pile_arxiv",
        "common_pile_pubmed",
        "proof_pile_2_arxiv",
    }
    assert sum(plan["downstream_partition"]["training_bytes"] for plan in plans) == 120_000_000
    assert sum(plan["downstream_partition"]["evaluation_bytes"] for plan in plans) == 12_000_000

    finepdfs = next(plan for plan in plans if plan["source"]["id"] == "finepdfs_edu_en")
    assert finepdfs["filters"]["required_language"] == "eng_Latn"
    assert finepdfs["filters"]["required_secondary_language"] == "eng_Latn"
    assert finepdfs["filters"]["reject_truncated"] is True
    assert finepdfs["filters"]["accepted_extractors"] == ["docling"]
    assert finepdfs["filters"]["minimum_science_term_hits"] == 3


def test_flagship_science_overlap_prioritizes_per_document_license_sources():
    path = ROOT / "research/flagship/science_cross_source_duplicates_v1.json"
    plan = validate_text_duplicate_config(json.loads(path.read_text()), config_dir=path.parent)

    assert [source["id"] for source in plan["sources"]] == [
        "common_pile_pubmed",
        "common_pile_arxiv",
        "pes2o_v3",
        "finepdfs_edu_en",
        "proof_pile_2_arxiv",
    ]
    assert sum(source["training_bytes"] for source in plan["sources"]) == 100_000_000
    assert sum(source["evaluation_bytes"] for source in plan["sources"]) == 10_000_000
    assert plan["policy"]["category"] == "science"


def test_flagship_science_contamination_reuses_frozen_payload_policy():
    path = ROOT / "research/flagship/science_contamination_v1.json"
    plan = validate_text_contamination_config(json.loads(path.read_text()), config_dir=path.parent)
    web_path = ROOT / "research/flagship/web_contamination_v1.json"
    web = validate_text_contamination_config(
        json.loads(web_path.read_text()), config_dir=web_path.parent
    )

    assert [source["id"] for source in plan["sources"]] == [
        "common_pile_pubmed",
        "common_pile_arxiv",
        "pes2o_v3",
        "finepdfs_edu_en",
        "proof_pile_2_arxiv",
    ]
    assert plan["policy"] == web["policy"]
    assert plan["benchmarks"] == web["benchmarks"]


def test_science_quota_failure_successors_only_increase_sample_margins():
    for source_id, expected in {
        "common_pile_arxiv": (22_000_000, 2_200_000),
        "proof_pile_2_arxiv": (8_000_000, 800_000),
    }.items():
        first_path = ROOT / f"research/flagship/science_{source_id}_v1.json"
        successor_path = ROOT / f"research/flagship/science_{source_id}_v2.json"
        first = validate_science_sample_config(
            json.loads(first_path.read_text()), config_dir=first_path.parent
        )
        successor = validate_science_sample_config(
            json.loads(successor_path.read_text()), config_dir=successor_path.parent
        )

        assert successor["source"] == first["source"]
        assert successor["rights"] == first["rights"]
        assert successor["filters"] == first["filters"]
        assert (
            successor["downstream_partition"]["training_bytes"],
            successor["downstream_partition"]["evaluation_bytes"],
        ) == expected
