import copy
import json
from pathlib import Path

import pytest

from speck.data import fineweb_edu_stock as web
from speck.data.acquisition_units import _unit_config

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def filters(monkeypatch):
    class Detector:
        def classify(self, text):
            return "en", 0.99

    monkeypatch.setattr(web, "_language_identifier", lambda: Detector())
    return json.loads(
        (ROOT / "archive/pregrant-history/research/flagship/web_fineweb_edu_v1.json").read_text()
    )["filters"]


def document():
    return {
        "content": "This passage explains how water changes temperature and how its measured properties vary under controlled conditions. "
        * 6,
        "metadata": {
            "id": "document-123",
            "url": "https://example.org/lesson",
            "language": "en",
            "language_score": 0.95,
            "quality_score": 3,
        },
    }


@pytest.mark.parametrize(
    "key,value,reason",
    [
        ("language_score", float("nan"), "language_score"),
        ("language_score", 1.1, "language_score"),
        ("language_score", True, "language_score"),
        ("quality_score", True, "quality_score"),
        ("quality_score", 2, "quality_score"),
        ("quality_score", 6, "quality_score"),
        ("id", "", "missing_document_id"),
        ("url", None, "invalid_url"),
        ("url", "https://porn.example/a", "excluded_host"),
    ],
)
def test_declared_metadata_requires_valid_values(filters, key, value, reason):
    doc = document()
    doc["metadata"][key] = value
    assert reason in web.fineweb_edu_rejection(doc, filters)[0]


def test_corpus_has_no_sampler_host_state_but_preserves_document_filters(filters, monkeypatch):
    filters["maximum_bytes_per_host"] = 1
    doc = document()
    assert web.fineweb_edu_rejection(doc, filters) == (None, 0.99)
    doc["content"] += " Contact person@private-domain.test for details."
    assert web.fineweb_edu_rejection(doc, filters)[0] == "web_qualified_PII"
    doc = document()
    doc["content"] = "repeat this line\n" * 100
    assert web.fineweb_edu_rejection(doc, filters)[0] == "web_duplicate_lines"


def test_registered_web_stock_binds_complete_view_and_unit_policy():
    plan = web.load_fineweb_edu_preparation(
        ROOT / "research/flagship/fineweb_edu_stock_preparation_v1.json"
    )
    assert len(plan["units"]) == 14
    assert plan["target_reference_tokens"] == 5280000000
    assert all(
        unit["reader"]["filters"] == {"language": "en", "min_score": 3} for unit in plan["units"]
    )
    assert plan["base"]["fineweb_edu_filters"]["maximum_bytes_per_host"] is None
    first = _unit_config(plan, plan["units"][0])
    changed = copy.deepcopy(plan)
    changed["base"]["fineweb_edu_filters"]["minimum_detected_English_probability"] = 0.9
    assert _unit_config(changed, plan["units"][0]) != first
    with pytest.raises(ValueError, match="only govern web"):
        _unit_config(plan, {**plan["units"][0], "category": "code"})
