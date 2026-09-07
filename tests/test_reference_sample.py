import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from speck.reference_sample import sample_reference_source, validate_reference_sample_config
from speck.stack_v3_refine import _sample_partition
from speck.text_near_duplicates import validate_text_duplicate_config

ROOT = Path(__file__).parents[1]


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


def test_initial_reference_plans_freeze_five_common_pile_margins():
    plans = []
    for path in sorted((ROOT / "research/flagship").glob("reference_*_v1.json")):
        raw = json.loads(path.read_text())
        if raw.get("format") == "speck_reference_sample":
            plans.append(validate_reference_sample_config(raw, config_dir=path.parent))

    initial = [plan for plan in plans if plan["source"]["id"] != "finewiki_en"]
    assert {plan["source"]["id"] for plan in initial} == {
        "common_pile_stackexchange",
        "common_pile_gutenberg",
        "common_pile_libretexts",
        "common_pile_oercommons",
        "common_pile_pressbooks",
    }
    assert sum(plan["downstream_partition"]["training_bytes"] for plan in initial) == 78_000_000
    assert sum(plan["downstream_partition"]["evaluation_bytes"] for plan in initial) == 7_800_000

    finewiki = next(plan for plan in plans if plan["source"]["id"] == "finewiki_en")
    assert finewiki["source"]["rows"] == 421_456
    assert finewiki["downstream_partition"]["training_bytes"] == 42_000_000
    assert finewiki["downstream_partition"]["evaluation_bytes"] == 4_200_000


def test_reference_failure_successors_only_correct_declared_source_treatment():
    first_path = ROOT / "research/flagship/reference_common_pile_gutenberg_v1.json"
    next_path = ROOT / "research/flagship/reference_common_pile_gutenberg_v2.json"
    first = validate_reference_sample_config(
        json.loads(first_path.read_text()), config_dir=first_path.parent
    )
    successor = validate_reference_sample_config(
        json.loads(next_path.read_text()), config_dir=next_path.parent
    )
    assert successor["source"]["fields"]["language"] == "metadata.language"
    assert first["source"]["fields"]["language"] is None
    assert successor["filters"] == first["filters"]
    assert successor["downstream_partition"] == first["downstream_partition"]

    first_path = ROOT / "research/flagship/reference_common_pile_oercommons_v1.json"
    next_path = ROOT / "research/flagship/reference_common_pile_oercommons_v2.json"
    first = validate_reference_sample_config(
        json.loads(first_path.read_text()), config_dir=first_path.parent
    )
    successor = validate_reference_sample_config(
        json.loads(next_path.read_text()), config_dir=next_path.parent
    )
    changed = {**first["filters"], "maximum_bytes_per_host": None}
    assert successor["filters"] == changed
    assert successor["source"] == first["source"]
    assert successor["downstream_partition"] == first["downstream_partition"]


def test_reference_overlap_prioritizes_public_domain_and_item_licensed_sources():
    path = ROOT / "research/flagship/reference_cross_source_duplicates_v1.json"
    plan = validate_text_duplicate_config(json.loads(path.read_text()), config_dir=path.parent)

    assert [source["id"] for source in plan["sources"]] == [
        "common_pile_gutenberg",
        "common_pile_oercommons",
        "common_pile_pressbooks",
        "common_pile_libretexts",
        "finewiki_en",
        "common_pile_stackexchange",
    ]
    assert sum(source["training_bytes"] for source in plan["sources"]) == 100_000_000
    assert sum(source["evaluation_bytes"] for source in plan["sources"]) == 10_000_000
    assert plan["policy"]["category"] == "reference"
