import gzip
import hashlib
import json

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
                "quality_score": None,
                "date": None,
                "source_partition": None,
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
