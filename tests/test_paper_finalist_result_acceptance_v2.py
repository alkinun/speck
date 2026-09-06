import json
from pathlib import Path

import pytest

from scripts.paper_finalist_continue import AUTOMATION, ordered_runs
from scripts.paper_finalist_result_acceptance_validate import (
    file_sha256,
)
from scripts.paper_finalist_result_acceptance_validate import (
    validate_program as validate_acceptance_v1,
)
from scripts.paper_finalist_result_acceptance_validate_v2 import validate_program
from speck.paper_finalist_analysis import collect_run_result
from tests.test_paper_finalist_source_coverage import complete_report

repository_root = Path(__file__).parents[1]
plan_path = repository_root / "research" / "paper-1" / "finalist_analysis_v2.json"
contract_path = repository_root / "research" / "paper-1" / "finalist_materialization_v1.json"
experiment = (
    repository_root
    / "experiments"
    / "Speck-Paper1-Finalist-131M"
    / "runs"
    / "pair-0-seed-42-order-0"
    / "dense_global_param_match"
)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_checkpoint(path):
    report = complete_report()
    expected = 23_496
    path.mkdir()
    (path / f"model_{expected:06d}.pt").write_bytes(b"model")
    (path / f"optimizer_{expected:06d}.pt").write_bytes(b"optimizer")
    metadata = {
        "step": expected,
        "global_tokens": 1_539_833_856,
        "partial": False,
        "manifest": "b84b09e0b701e35d84487cf6f91e6da9c9fb686b7f6efe67b2e2f5f301fda98e",
        "validation_step": expected,
        "validation_tokens": 19_988_480,
        "validation_history": report["validation_history"],
        "peak_allocated_bytes": 123,
        "resolved": {
            "seed": 42,
            "data_token_offset": 0,
            "train_tokens": 1_539_833_856,
            "batch_tokens": 65_536,
            "sequence_length": 4_096,
            "parameters": 153_977_088,
            "world_size": 1,
        },
    }
    atomic_json(path / f"metadata_{expected:06d}.json", metadata)
    atomic_json(
        path / f"timing_{expected:06d}.json",
        {"optimizer_seconds": 43.0, "steady_training_seconds": 40.0},
    )
    (path / f"complete_{expected:06d}").write_text("complete\n", encoding="utf-8")
    atomic_json(
        path / "run_summary.json",
        {
            "partial": False,
            "completed_steps": expected,
            "global_tokens": 1_539_833_856,
            "validation_history": report["validation_history"],
        },
    )


def one_control_state(tmp_path):
    run_name = ordered_runs()[0]
    checkpoint = tmp_path / "checkpoint"
    write_checkpoint(checkpoint)
    report = collect_run_result(plan_path, contract_path, experiment, checkpoint)
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
            "result": {"path": result_path.relative_to(tmp_path).as_posix(), "sha256": result_hash},
            "next_run": ordered_runs()[1],
            "quality_dependent_branching": False,
            "trigger_disabled_before_collection": True,
            "polling": False,
            "automatic_retry": False,
            "automation_contract_sha256": file_sha256(AUTOMATION),
        },
    )
    qualification_path = tmp_path / "qualification.json"
    atomic_json(
        qualification_path,
        {
            "status": (
                "materialization_data_and_storage_qualified_runtime_analysis_and_release_gates_blocked"
            ),
            "runs": [
                {
                    "run": run_name,
                    "experiment": experiment.relative_to(repository_root).as_posix(),
                    "checkpoint_directory": str(checkpoint),
                }
            ],
        },
    )
    program = {
        "finalist_qualification": {
            "result": qualification_path.relative_to(tmp_path).as_posix(),
            "sha256": file_sha256(qualification_path),
            "status": (
                "materialization_data_and_storage_qualified_runtime_analysis_and_release_gates_blocked"
            ),
        },
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
        },
    }
    program_path = tmp_path / "program.json"
    atomic_json(program_path, program)
    return program_path, result_path, transition_path, checkpoint


def rehash_result(program_path, result_path, transition_path):
    result_hash = file_sha256(result_path)
    program = json.loads(program_path.read_text(encoding="utf-8"))
    program["finalist_evidence"]["control_results"][0]["sha256"] = result_hash
    atomic_json(program_path, program)
    transition = json.loads(transition_path.read_text(encoding="utf-8"))
    transition["result"]["sha256"] = result_hash
    atomic_json(transition_path, transition)


def test_v2_accepts_exact_replay_from_retained_checkpoint(tmp_path):
    program, _, _, _ = one_control_state(tmp_path)
    result = validate_program(program, repository_root, tmp_path)
    assert result["accepted_results"] == 1
    assert result["retained_checkpoint_provenance"] == 1
    assert result["stable_result_replay"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("analysis_plan_sha256", "0" * 64),
        ("experiment", "/tmp/unqualified-experiment"),
    ),
)
def test_v2_rejects_provenance_fields_that_v1_accepts(tmp_path, field, value):
    program, result_path, transition_path, _ = one_control_state(tmp_path)
    report = json.loads(result_path.read_text(encoding="utf-8"))
    report[field] = value
    atomic_json(result_path, report)
    rehash_result(program, result_path, transition_path)
    assert validate_acceptance_v1(program, repository_root, tmp_path)["status"] == "valid"
    with pytest.raises(ValueError, match="provenance|replay"):
        validate_program(program, repository_root, tmp_path)


def test_v2_rejects_claimed_checkpoint_hash_that_v1_accepts(tmp_path):
    program, result_path, transition_path, _ = one_control_state(tmp_path)
    report = json.loads(result_path.read_text(encoding="utf-8"))
    report["checkpoint"]["model_sha256"] = "0" * 64
    atomic_json(result_path, report)
    rehash_result(program, result_path, transition_path)
    assert validate_acceptance_v1(program, repository_root, tmp_path)["status"] == "valid"
    with pytest.raises(ValueError, match="replay"):
        validate_program(program, repository_root, tmp_path)


def test_v2_rejects_checkpoint_changed_after_collection_that_v1_accepts(tmp_path):
    program, _, _, checkpoint = one_control_state(tmp_path)
    (checkpoint / "model_023496.pt").write_bytes(b"changed model")
    assert validate_acceptance_v1(program, repository_root, tmp_path)["status"] == "valid"
    with pytest.raises(ValueError, match="replay"):
        validate_program(program, repository_root, tmp_path)
