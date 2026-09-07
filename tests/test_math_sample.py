import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from speck.math_sample import (
    _jsonl_zstd_batches,
    _language_result,
    sample_math_source,
    validate_math_sample_config,
)
from speck.stack_v3_refine import _sample_partition

ROOT = Path(__file__).parents[1]


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_for(split):
    settings = {
        "seed": 42,
        "category": "math",
        "modulus": 2,
        "evaluation_remainders": [0],
    }
    for value in range(100):
        text = (
            f"English proof {value}: let $x^2 + y^2 = z^2$. We explain every algebraic step "
            "and preserve the equation exactly for mathematical study."
        )
        if _sample_partition({"text": text}, settings) == split:
            return text
    raise AssertionError("could not construct math partition fixture")


def _config(path, output):
    return {
        "format": "speck_math_sample",
        "format_version": 1,
        "status": "bounded_sample_authorized_not_training_authority",
        "source": {
            "id": "example_math",
            "repo": "example/math",
            "revision": "a" * 40,
            "official_url": "https://example.com/math",
            "path": str(path),
            "sha256": _sha256(path),
            "rows": 3,
            "file_format": "parquet",
            "text_field": "text",
            "metadata_json_field": None,
            "fields": {
                "document_id": "id",
                "url": "url",
                "language": "language",
                "language_score": "language_score",
                "quality_score": "quality",
                "code_language": None,
                "detected_licenses": None,
                "license_type": None,
                "repository": None,
                "file_path": None,
            },
        },
        "rights": {
            "dataset_license": "test",
            "upstream_terms": "test",
            "authority": "manual_review_required",
            "redistribution": "undecided",
        },
        "filters": {
            "language_policy": "metadata_and_math_prose",
            "required_language": "en",
            "minimum_language_score": 0.8,
            "minimum_quality_score": 4,
            "minimum_prose_alphabetic_characters": 20,
            "minimum_detected_English_probability": 0.8,
            "min_document_bytes": 20,
            "max_document_bytes": 10_000,
            "maximum_duplicate_line_ratio": 0.5,
            "maximum_bytes_per_host": 10_000,
            "allowed_email_placeholders": [],
            "allowed_ipv4_placeholders": [],
            "language_detector": "py3langid==0.3.0",
            "accepted_detected_licenses": [],
            "required_license_type": None,
        },
        "downstream_partition": {
            "seed": 42,
            "category": "math",
            "modulus": 2,
            "evaluation_remainders": [0],
            "training_bytes": 1,
            "evaluation_bytes": 1,
        },
        "output_directory": str(output),
    }


def test_math_sample_preserves_latex_and_filters_metadata(tmp_path):
    train = _text_for("train")
    evaluation = _text_for("eval")
    rows = [
        {
            "id": "train",
            "text": train,
            "url": "https://math.example/train",
            "language": "en",
            "language_score": 0.99,
            "quality": 5,
        },
        {
            "id": "eval",
            "text": evaluation,
            "url": "https://math.example/eval",
            "language": "en",
            "language_score": 0.99,
            "quality": 4,
        },
        {
            "id": "low",
            "text": "A low quality English math page with $x=1$.",
            "url": "https://math.example/low",
            "language": "en",
            "language_score": 0.99,
            "quality": 3,
        },
    ]
    path = tmp_path / "math.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    report = sample_math_source(validate_math_sample_config(_config(path, tmp_path / "output")))

    retained = [
        json.loads(line)
        for line in (tmp_path / "output/tokenizer-input.jsonl").read_text().splitlines()
    ]
    assert {record["text"] for record in retained} == {train, evaluation}
    assert all("$x^2 + y^2 = z^2$" in record["text"] for record in retained)
    assert report["counts"]["records_with_latex_signal"] == 2
    assert report["downstream_partition"]["observed"]["train_bytes"] >= 1
    assert report["downstream_partition"]["observed"]["eval_bytes"] >= 1
    assert report["gates"]["training_authority"] == "blocked"


def test_math_code_policy_exempts_notation_but_rejects_non_english_comments(tmp_path):
    path = tmp_path / "unused"
    path.write_bytes(b"fixture")
    raw = _config(path, tmp_path / "output")
    raw["source"]["fields"]["code_language"] = "language"
    raw["filters"]["language_policy"] = "code_comments_and_docstrings"
    raw["filters"]["required_language"] = None
    raw["filters"]["minimum_language_score"] = None
    raw["filters"]["minimum_quality_score"] = None
    filters = validate_math_sample_config(raw)["filters"]

    assert _language_result("theorem identity : x = x := rfl", "Lean", filters)[0] == (
        "insufficient_prose"
    )
    french = (
        "// Ceci est une explication mathématique détaillée écrite entièrement en français. "
        "Elle décrit soigneusement chaque étape du calcul et fournit une démonstration complète."
    )
    assert _language_result(french, "C++", filters)[0] == "non_English"


def test_math_sample_reads_jsonl_zstd_without_rewriting_rows(tmp_path):
    path = tmp_path / "math.jsonl.zst"
    rows = [{"text": "theorem x"}, {"text": "proof y"}]
    with pa.output_stream(path, compression="zstd") as handle:
        handle.write("".join(json.dumps(row) + "\n" for row in rows).encode())

    assert _jsonl_zstd_batches(path, batch_size=1) == [[rows[0]], [rows[1]]]


def test_flagship_math_sample_plans_freeze_six_source_margins():
    plans = []
    for path in sorted((ROOT / "research/flagship").glob("math_*_v1.json")):
        if path.name == "math_megamath_code_v1.json":
            continue
        plans.append(
            validate_math_sample_config(json.loads(path.read_text()), config_dir=path.parent)
        )

    assert {plan["source"]["id"] for plan in plans} == {
        "finemath_4plus",
        "infiwebmath_4plus",
        "megamath_web_pro",
        "openwebmath",
        "proof_pile_2_algebraic_stack",
    }
    assert sum(plan["downstream_partition"]["training_bytes"] for plan in plans) == 114_000_000
    assert sum(plan["downstream_partition"]["evaluation_bytes"] for plan in plans) == 11_400_000
