import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

_PATH = Path(__file__).parents[2] / "experiments/corpus-audit/audit_cohort_similarity.py"
_SPEC = importlib.util.spec_from_file_location("audit_cohort_similarity", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def fixture(texts):
    rows = []
    assessments = []
    for index, text in enumerate(texts):
        row = {
            "id": str(index),
            "cohort": "test",
            "text": text,
            "consumed_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "tokens": len(text.split()),
            "sampling": {"weight": 1},
        }
        rows.append(row)
        assessments.append(
            {
                **row,
                "candidate_partition": "quarantine" if index == 0 else "train",
                "family_component_sha256": str(index),
                "training_admitted": False,
            }
        )
    return rows, assessments


def test_detects_exact_near_and_embedded_copy_without_partition_changes():
    original = " ".join(f"word{i}" for i in range(200))
    texts = [
        original,
        original,
        original.replace("word100", "changed"),
        " ".join(f"prefix{i}" for i in range(300)) + " " + original,
        " ".join(f"unrelated{i}" for i in range(200)),
    ]
    rows, assessments = fixture(texts)
    before = json.dumps(assessments)
    result = _MODULE.analyze(rows, assessments)
    assert result["pairs_compared"] == 10
    assert len(result["matches"]) == 6
    assert sum(pair["exact"] for pair in result["matches"]) == 1
    assert result["prior_held_records"] == 1
    assert json.dumps(assessments) == before
    assert _MODULE.analyze(rows[::-1], assessments[::-1]) == result


def test_shared_ends_do_not_hide_a_different_middle():
    prefix = " ".join(f"prefix{i}" for i in range(100))
    suffix = " ".join(f"suffix{i}" for i in range(100))
    texts = [
        prefix + " " + " ".join(f"{label}{i}" for i in range(3000)) + " " + suffix
        for label in ("first", "second")
    ]
    assert _MODULE.analyze(*fixture(texts))["matches"] == []


def test_small_boilerplate_is_not_a_near_copy():
    assert _MODULE.analyze(*fixture(["hello world", "hello  world"]))["matches"] == []


@pytest.mark.parametrize("mutation", ["text", "tokens", "sampling", "ids"])
def test_rejects_changed_inputs(mutation):
    rows, assessments = fixture(["some source", "other source"])
    if mutation == "text":
        rows[0]["text"] = "changed"
    elif mutation == "tokens":
        rows[0]["tokens"] += 1
    elif mutation == "sampling":
        rows[0]["sampling"] = {"weight": 2}
    else:
        rows.append(rows[0])
    with pytest.raises(ValueError):
        _MODULE.analyze(rows, assessments)
