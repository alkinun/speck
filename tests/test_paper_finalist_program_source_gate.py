import json
from pathlib import Path

import pytest

from speck.paper import FINALIST_SOURCE_IDS, _file_sha256, _validate_finalist_source_losses


def complete_losses():
    return {source: 3.0 + index * 0.001 for index, source in enumerate(FINALIST_SOURCE_IDS)}


def test_finalist_program_source_gate_accepts_complete_finite_set():
    _validate_finalist_source_losses(complete_losses())


def test_finalist_program_source_gate_rejects_missing_source():
    values = complete_losses()
    values.pop("dclm")
    with pytest.raises(ValueError, match="every expected"):
        _validate_finalist_source_losses(values)


def test_finalist_program_source_gate_rejects_extra_source():
    values = complete_losses()
    values["unexpected"] = 3.0
    with pytest.raises(ValueError, match="every expected"):
        _validate_finalist_source_losses(values)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf"), True))
def test_finalist_program_source_gate_rejects_non_finite_or_boolean(value):
    values = complete_losses()
    values["dclm"] = value
    with pytest.raises(ValueError, match="non-finite"):
        _validate_finalist_source_losses(values)


def test_finalist_program_source_gate_qualification_is_bound():
    root = Path(__file__).parents[1]
    path = root / "results" / "Speck-Paper1" / "finalist-program-source-gate-v1.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["status"] == (
        "complete_finite_11_source_gate_wired_precommit_without_scientific_change"
    )
    assert set(result["expected_sources"]) == FINALIST_SOURCE_IDS
    assert result["implementation"]["program_validator"]["sha256"] == _file_sha256(
        root / "speck" / "paper.py"
    )
    assert result["finalizer_integration"]["failure_before_commit"] is True
    assert result["finalizer_integration"]["failure_before_successor_schedule"] is True
    assert result["decision"]["runner_analyzer_plan_or_threshold_changed"] is False
