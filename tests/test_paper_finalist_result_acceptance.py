import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.paper_finalist_continue import AUTOMATION, ordered_runs
from scripts.paper_finalist_result_acceptance_validate import file_sha256, validate_program
from scripts.paper_finalist_source_coverage_validate import EXPECTED_SOURCES
from tests.test_paper_finalist_source_coverage import complete_report

repository_root = Path(__file__).parents[1]


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def one_control_state(tmp_path):
    run_name = ordered_runs()[0]
    report = complete_report()
    report["run"] = run_name
    report["arm_id"] = "dense_global_param_match"
    result_path = tmp_path / "results" / "Speck-Paper1" / "finalist-runs" / f"{run_name}.json"
    atomic_json(result_path, report)
    result_hash = file_sha256(result_path)
    transition_path = (
        tmp_path / "results" / "Speck-Paper1" / "finalist-transitions" / f"{run_name}.json"
    )
    atomic_json(
        transition_path,
        {
            "format": "speck_paper_finalist_automatic_transition",
            "format_version": 2,
            "status": "complete",
            "completed_run": run_name,
            "result": {
                "path": result_path.relative_to(tmp_path).as_posix(),
                "sha256": result_hash,
            },
            "next_run": ordered_runs()[1],
            "quality_dependent_branching": False,
            "trigger_disabled_before_collection": True,
            "polling": False,
            "automatic_retry": False,
            "automation_contract_sha256": file_sha256(AUTOMATION),
        },
    )
    program = {
        "finalist_evidence": {
            "status": "in_progress",
            "control_results": [
                {
                    "pair": 0,
                    "path": result_path.relative_to(tmp_path).as_posix(),
                    "sha256": result_hash,
                    "status": "complete_qualified",
                }
            ],
            "candidate_results": [],
            "time_to_quality_target": None,
            "analysis_result": None,
            "next_run": ordered_runs()[1],
            "failed_attempts": [
                {
                    "attempt": 1,
                    "status": "operator_interrupted_after_step_1_no_checkpoint_or_result",
                }
            ],
            "active_rerun": {
                "attempt": 2,
                "status": "frozen_after_operator_interruption_before_identical_restart",
            },
        }
    }
    program_path = tmp_path / "program.json"
    atomic_json(program_path, program)
    return program_path, result_path, transition_path


def test_finalist_acceptance_validates_one_complete_control(tmp_path):
    program, _, _ = one_control_state(tmp_path)
    assert validate_program(program, repository_root, tmp_path) == {
        "status": "valid",
        "accepted_results": 1,
        "controls": 1,
        "candidates": 0,
        "next_run": ordered_runs()[1],
        "target_locked": False,
        "analysis_complete": False,
        "complete_source_coverage": True,
    }


def test_finalist_acceptance_rejects_reference_hash_drift(tmp_path):
    program_path, _, _ = one_control_state(tmp_path)
    program = json.loads(program_path.read_text(encoding="utf-8"))
    program["finalist_evidence"]["control_results"][0]["sha256"] = "0" * 64
    atomic_json(program_path, program)
    with pytest.raises(ValueError, match="result reference"):
        validate_program(program_path, repository_root, tmp_path)


def test_finalist_acceptance_rejects_transition_edge_drift(tmp_path):
    program_path, _, transition_path = one_control_state(tmp_path)
    transition = json.loads(transition_path.read_text(encoding="utf-8"))
    transition["next_run"] = ordered_runs()[2]
    atomic_json(transition_path, transition)
    with pytest.raises(ValueError, match="transition is invalid"):
        validate_program(program_path, repository_root, tmp_path)


def test_finalist_acceptance_rejects_incomplete_source_set(tmp_path):
    program_path, result_path, _ = one_control_state(tmp_path)
    report = json.loads(result_path.read_text(encoding="utf-8"))
    for entry in report["validation_history"]:
        entry["validation_source_losses"].pop(EXPECTED_SOURCES[-1])
    report["final_validation"] = report["validation_history"][-1]
    atomic_json(result_path, report)
    program = json.loads(program_path.read_text(encoding="utf-8"))
    new_hash = file_sha256(result_path)
    program["finalist_evidence"]["control_results"][0]["sha256"] = new_hash
    atomic_json(program_path, program)
    with pytest.raises(ValueError, match="every expected source"):
        validate_program(program_path, repository_root, tmp_path)


def test_finalist_acceptance_rejects_lost_failure_history(tmp_path):
    program_path, _, _ = one_control_state(tmp_path)
    program = deepcopy(json.loads(program_path.read_text(encoding="utf-8")))
    program["finalist_evidence"]["failed_attempts"] = []
    atomic_json(program_path, program)
    with pytest.raises(ValueError, match="lost the failed-attempt"):
        validate_program(program_path, repository_root, tmp_path)


def test_finalist_acceptance_qualification_is_bound_to_verifier():
    qualification_path = (
        repository_root
        / "results"
        / "Speck-Paper1"
        / "finalist-result-acceptance-qualified-v1.json"
    )
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    verifier = qualification["implementation"]["verifier"]
    sidecar = qualification["implementation"]["source_sidecar"]
    assert qualification["status"] == "qualified_before_first_accepted_result_append_only_not_wired"
    assert file_sha256(repository_root / verifier["path"]) == verifier["sha256"]
    assert file_sha256(repository_root / sidecar["path"]) == sidecar["sha256"]
    assert qualification["decision"]["must_run_after_every_automatic_result_commit"] is True
    assert qualification["decision"]["wired_into_frozen_automation"] is False
    assert qualification["decision"]["active_runner_or_program_modified"] is False
