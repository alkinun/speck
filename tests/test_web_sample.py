import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from speck.stack_v3_refine import _sample_partition
from speck.text_gitleaks_filter import validate_text_gitleaks_config
from speck.text_near_duplicates import (
    analyze_text_cross_source_duplicates,
    validate_text_duplicate_config,
)
from speck.web_sample import (
    _host,
    _raw_pii,
    sample_web_source,
    validate_web_sample_config,
)

ROOT = Path(__file__).parents[1]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _text_for(split):
    settings = {
        "seed": 42,
        "category": "web",
        "modulus": 2,
        "evaluation_remainders": [0],
    }
    for index in range(100):
        text = (
            f"Educational English document {index}. The history of scientific discovery "
            "includes careful observation, detailed experiments, and open discussion among "
            "researchers. This article explains why evidence matters and how people can evaluate "
            "competing ideas. " * 8
        )
        if _sample_partition({"text": text}, settings) == split:
            return text
    raise AssertionError("could not construct split fixture")


def _web_config(path, output):
    return {
        "format": "speck_web_sample",
        "format_version": 1,
        "status": "bounded_sample_authorized_not_training_authority",
        "source": {
            "id": "example_web",
            "repo": "example/web",
            "revision": "a" * 40,
            "official_url": "https://example.com/dataset",
            "path": str(path),
            "sha256": _sha256(path),
            "rows": 4,
            "content_column": "text",
            "metadata_json_column": None,
            "fields": {
                "document_id": "id",
                "url": "url",
                "language": "language",
                "language_score": "language_score",
                "quality_score": None,
            },
        },
        "rights": {
            "dataset_license": "test",
            "upstream_terms": "test",
            "authority": "manual_review_required",
        },
        "filters": {
            "required_language": "en",
            "minimum_language_score": 0.8,
            "minimum_quality_score": None,
            "min_document_bytes": 10,
            "max_document_bytes": 10_000,
            "minimum_alphabetic_ratio": 0.2,
            "maximum_duplicate_line_ratio": 0.5,
            "maximum_bytes_per_host": 10_000,
            "adult_host_terms": ["sexdating"],
            "allowed_email_placeholders": ["email@example.com"],
            "allowed_ipv4_placeholders": [],
            "language_detector": "py3langid==0.3.0",
            "minimum_detected_English_probability": 0.8,
        },
        "downstream_partition": {
            "seed": 42,
            "category": "web",
            "modulus": 2,
            "evaluation_remainders": [0],
            "training_bytes": 1,
            "evaluation_bytes": 1,
        },
        "output_directory": str(output),
    }


def test_web_sample_filters_pii_and_adult_host_and_fills_both_splits(tmp_path):
    rows = [
        {
            "text": _text_for("train"),
            "id": "train",
            "url": "https://good.example/train",
            "language": "en",
            "language_score": 0.99,
        },
        {
            "text": _text_for("eval"),
            "id": "eval",
            "url": "https://good.example/eval",
            "language": "en",
            "language_score": 0.99,
        },
        {
            "text": "Contact private.person@unknown.example. " + "English prose. " * 40,
            "id": "email",
            "url": "https://other.example/email",
            "language": "en",
            "language_score": 0.99,
        },
        {
            "text": "A long English page. " * 40,
            "id": "adult",
            "url": "https://sexdating.example/page",
            "language": "en",
            "language_score": 0.99,
        },
    ]
    path = tmp_path / "web.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    config = validate_web_sample_config(_web_config(path, tmp_path / "output"))
    report = sample_web_source(config)

    assert report["status"] == "bounded_sample_complete_not_training_authority"
    assert report["counts"]["records_sampled"] == 2
    assert report["downstream_partition"]["observed"]["train_records"] == 1
    assert report["downstream_partition"]["observed"]["eval_records"] == 1
    assert _raw_pii("private.person@unknown.example", set(), set())[0]
    assert "sexdating" in _host("https://sexdating.example/page")


def test_text_gitleaks_contract_accepts_web_and_rejects_code(tmp_path):
    raw = {
        "format": "speck_text_gitleaks_filter",
        "format_version": 1,
        "status": "exclusion_authorized_not_training_authority",
        "input": {
            "path": str(tmp_path / "input"),
            "sha256": "a" * 64,
            "parent_report": str(tmp_path / "parent"),
            "parent_report_sha256": "b" * 64,
            "parent_format": "example",
            "parent_status": "example",
        },
        "scanner": {
            "name": "gitleaks",
            "version": "1",
            "official_url": "https://example.com",
            "binary": str(tmp_path / "binary"),
            "binary_sha256": "c" * 64,
            "release_archive_sha256": "d" * 64,
            "report": str(tmp_path / "report"),
            "report_sha256": "e" * 64,
            "redaction_percent": 100,
        },
        "downstream_partition": {
            "seed": 42,
            "category": "web",
            "modulus": 10,
            "evaluation_remainders": [0],
            "training_bytes": 1,
            "evaluation_bytes": 1,
        },
        "output_directory": str(tmp_path / "output"),
    }
    assert validate_text_gitleaks_config(raw)["downstream_partition"]["category"] == "web"
    raw["downstream_partition"]["category"] = "code"
    with pytest.raises(ValueError, match="supported text category"):
        validate_text_gitleaks_config(raw)


def _record(text, host):
    return {
        "text": text,
        "released_content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "host": host,
    }


def test_text_overlap_removes_exact_cross_source_match_with_web_partition(tmp_path):
    shared = _record("Shared English document. " * 30, "shared.example")
    clean = _record("Independent English document. " * 30, "clean.example")
    sources = []
    for index, records in enumerate(([shared], [shared, clean]), 1):
        path = tmp_path / f"source-{index}.jsonl"
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
        parent = {
            "format": "parent_result",
            "status": "complete_not_training_authority",
            "outputs": {"tokenizer_input": {"sha256": _sha256(path)}},
            "gates": {"training_authority": "blocked"},
        }
        parent_path = tmp_path / f"parent-{index}.json"
        parent_path.write_text(json.dumps(parent))
        sources.append(
            {
                "id": f"source_{index}",
                "precedence": index,
                "path": str(path),
                "sha256": _sha256(path),
                "parent_report": str(parent_path),
                "parent_report_sha256": _sha256(parent_path),
                "parent_format": parent["format"],
                "parent_status": parent["status"],
                "training_bytes": 0,
                "evaluation_bytes": 0,
            }
        )
    config = validate_text_duplicate_config(
        {
            "format": "speck_text_cross_source_duplicates",
            "format_version": 1,
            "status": "analysis_authorized_not_training_authority",
            "sources": sources,
            "policy": {
                "category": "web",
                "normalization": "NFKC+lower+lexical-code-tokens",
                "token_pattern": "[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[^\\s]",
                "shingle_tokens": 3,
                "minimum_document_tokens": 3,
                "maximum_document_tokens": 1000,
                "num_perm": 16,
                "minhash_seed": 42,
                "lsh_threshold": 0.8,
                "verified_jaccard_threshold": 0.8,
                "partition_seed": 42,
                "partition_modulus": 10,
                "partition_evaluation_remainders": [0],
            },
            "output_directory": str(tmp_path / "dedup"),
        }
    )
    report = analyze_text_cross_source_duplicates(config)

    assert report["policy"]["category"] == "web"
    assert len(report["exact_cross_source_matches"]) == 1
    assert report["sources"][1]["counts"]["records_retained"] == 1
    assert report["gates"]["benchmark_contamination"] == "pending_evaluation_firewall"


def test_recorded_web_sources_bind_implementation_and_preserve_scope():
    result = json.loads(
        (ROOT / "results/data/web-tokenizer-sources-20260907.json").read_text()
    )
    assert result["status"] == (
        "bounded_web_sampling_security_overlap_pass_training_authority_blocked"
    )
    for path, digest in result["implementation"].values():
        assert digest == _sha256(ROOT / path)
    for path, digest in result["configs"].values():
        assert digest == _sha256(ROOT / path)
    for source in result["sources"].values():
        assert source["final_training_bytes"] >= source["training_target_bytes"]
        assert source["final_evaluation_bytes"] >= source["evaluation_target_bytes"]
    overlap = result["overlap"]
    assert overlap["retained_documents"] == 31_092
    assert overlap["exact_cross_source_matches"] == 0
    assert overlap["verified_near_cross_source_matches"] == 0
    assert "not claimed disjoint" in overlap["scope_limit"]
    assert any(
        gate.startswith("evaluation-firewall benchmark contamination")
        for gate in result["blocked_gates"]
    )
