import hashlib

import pyarrow as pa
import pyarrow.parquet as pq

from speck.reference_sample import sample_reference_source, validate_reference_sample_config
from speck.stack_v3_refine import _sample_partition


def _text_for(split):
    settings = {"seed": 42, "category": "reference", "modulus": 2, "evaluation_remainders": [0]}
    for index in range(100):
        text = f"Reference article {index} explains history, terminology, evidence, and practical examples in clear English prose."
        if _sample_partition({"text": text}, settings) == split:
            return text
    raise AssertionError("partition fixture failed")


def test_reference_wrapper_preserves_category_and_result_identity(tmp_path):
    rows = [{"id": "train", "text": _text_for("train")}, {"id": "eval", "text": _text_for("eval")}]
    path = tmp_path / "reference.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    raw = {
        "format": "speck_reference_sample",
        "format_version": 1,
        "status": "bounded_sample_authorized_not_training_authority",
        "source": {
            "id": "example_reference",
            "repo": "example/reference",
            "revision": "a" * 40,
            "official_url": "https://example.com",
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": 2,
            "file_format": "parquet",
            "fields": {
                "text": "text",
                "document_id": "id",
                "title": None,
                "url": None,
                "license": None,
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
            "maximum_spaced_OCR_sequences": 4,
            "maximum_replacement_character_ratio": 0.001,
            "boilerplate_line_terms": ["cookie policy"],
            "maximum_boilerplate_line_ratio": 0.5,
            "accepted_document_licenses": [],
            "minimum_science_term_hits": 0,
            "science_terms": ["reference"],
            "maximum_bytes_per_host": None,
            "allowed_email_placeholders": [],
            "allowed_ipv4_placeholders": [],
        },
        "downstream_partition": {
            "seed": 42,
            "category": "reference",
            "modulus": 2,
            "evaluation_remainders": [0],
            "training_bytes": 1,
            "evaluation_bytes": 1,
        },
        "output_directory": str(tmp_path / "output"),
    }
    config = validate_reference_sample_config(raw)
    report = sample_reference_source(config)

    assert report["format"] == "speck_reference_sample_result"
    assert report["plan_fingerprint"] == config["plan_fingerprint"]
    assert report["downstream_partition"]["category"] == "reference"
    assert report["gates"]["English_and_reference_content"] == "pass"
    assert report["gates"]["training_authority"] == "blocked"
