import json
from pathlib import Path

import pytest

from speck.data.finemath_stock import finemath_rejection, load_finemath_preparation
from speck.data.source_stock import domain_concentration


def filters():
    root = Path(__file__).resolve().parents[2]
    return json.loads(
        (
            root / "archive/pregrant-history/research/flagship/math_finemath_4plus_v1.json"
        ).read_text()
    )["filters"]


@pytest.mark.parametrize("score", [None, True, float("nan"), float("inf"), 0.79, 1.1])
def test_finemath_requires_valid_released_language_confidence(score):
    document = {
        "content": "Valid English scientific prose. " * 20,
        "metadata": {"language_score": score},
    }
    assert finemath_rejection(document, filters()) == "finemath_metadata_language_score"


def test_domain_report_keeps_missing_hosts_separate_without_reweighting(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in [
                {"text": "aaaa", "host": "one.example"},
                {"text": "bb", "host": "two.example"},
                {"text": "cc", "host": "two.example"},
                {"text": "λ", "host": None},
            ]
        )
    )
    report = domain_concentration(path)
    assert report["known_host_utf8_bytes"] == 8
    assert report["unknown_host_utf8_bytes"] == 2
    assert report["known_host_byte_hhi"] == 0.5
    assert report["known_host_documents"] == 3
    assert report["unknown_host_documents"] == 1


def test_finemath_plan_keeps_grade_and_explicit_corpus_policy():
    root = Path(__file__).resolve().parents[2]
    plan = load_finemath_preparation(root / "research/flagship/finemath_stock_preparation_v1.json")
    assert len(plan["units"]) == 8
    assert sum(unit["stop_row"] for unit in plan["units"]) == 837440
    assert all(
        unit["reader"]["filters"] == {"language": "en", "min_score": 4} for unit in plan["units"]
    )
    assert plan["base"]["finemath_filters"]["maximum_bytes_per_host"] is None
    assert plan["report_domain_concentration"] is True
