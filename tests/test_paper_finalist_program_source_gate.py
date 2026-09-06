import json
from pathlib import Path

import pytest

from scripts.paper_finalist_continue import ordered_runs
from speck.paper import (
    FINALIST_SOURCE_IDS,
    _file_sha256,
    _validate_finalist_evidence,
    _validate_finalist_source_losses,
)


def complete_losses():
    return {source: 3.0 + index * 0.001 for index, source in enumerate(FINALIST_SOURCE_IDS)}


def one_control_fixture(tmp_path, *, missing_source=False):
    order = ordered_runs()
    automation_path = tmp_path / "automation.json"
    automation_path.write_text(json.dumps({"execution_order": order}), encoding="utf-8")
    failed_path = tmp_path / "failed.json"
    rerun_path = tmp_path / "rerun.json"
    failed_path.write_text("{}", encoding="utf-8")
    rerun_path.write_text("{}", encoding="utf-8")
    history = []
    for step in (0, 5874, 11748, 17622, 23496):
        source_losses = complete_losses()
        if missing_source:
            source_losses.pop("dclm")
        history.append(
            {
                "step": step,
                "validation_source_losses": source_losses,
            }
        )
    result = {
        "format": "speck_paper_finalist_run_result",
        "format_version": 2,
        "status": "complete_qualified",
        "run": order[0],
        "arm_id": "dense_global_param_match",
        "pair": {"pair": 0},
        "training_tokens": 1_539_833_856,
        "non_finite_steps": 0,
        "validation_history": history,
        "final_validation": history[-1],
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    evidence = {
        "status": "in_progress",
        "control_results": [
            {
                "pair": 0,
                "path": "result.json",
                "sha256": _file_sha256(result_path),
                "status": "complete_qualified",
            }
        ],
        "candidate_results": [],
        "time_to_quality_target": None,
        "analysis_result": None,
        "next_run": order[1],
        "failed_attempts": [{"path": "failed.json", "sha256": _file_sha256(failed_path)}],
        "active_rerun": {"path": "rerun.json", "sha256": _file_sha256(rerun_path)},
    }
    return evidence, {"contract": "automation.json"}


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


def test_finalist_program_integration_accepts_complete_control(tmp_path):
    evidence, automation = one_control_fixture(tmp_path)
    _validate_finalist_evidence(evidence, automation, tmp_path)


def test_finalist_program_integration_rejects_missing_source(tmp_path):
    evidence, automation = one_control_fixture(tmp_path, missing_source=True)
    with pytest.raises(ValueError, match="every expected validation source"):
        _validate_finalist_evidence(evidence, automation, tmp_path)


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
