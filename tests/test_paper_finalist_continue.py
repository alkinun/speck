import json
from pathlib import Path

import pytest

import scripts.paper_finalist_continue as continuation
from scripts.paper_finalist_continue import (
    arguments,
    candidate_runs,
    control_runs,
    expected_next,
    ordered_runs,
)


def evidence(controls=0, candidates=0, target=False, analysis=False):
    return {
        "control_results": [{"pair": pair} for pair in range(controls)],
        "candidate_results": [{"pair": pair} for pair in range(candidates)],
        "time_to_quality_target": {"status": "locked"} if target else None,
        "analysis_result": {"status": "complete"} if analysis else None,
    }


def test_finalist_continue_arguments():
    run = control_runs()[0]
    args = arguments(["launch", run])
    assert args.command == "launch"
    assert args.run == run
    final = arguments(
        ["finalize", run, "--training-unit", "train.service", "--trigger-unit", "path.path"]
    )
    assert final.training_unit == "train.service"
    assert final.trigger_unit == "path.path"


def test_finalist_order_is_all_controls_then_all_candidates():
    assert ordered_runs() == control_runs() + candidate_runs()
    assert len(control_runs()) == len(candidate_runs()) == 6


def test_finalist_next_state_is_control_first_then_candidate():
    assert expected_next(evidence()) == control_runs()[0]
    assert expected_next(evidence(controls=5)) == control_runs()[5]
    assert expected_next(evidence(controls=6, target=True)) == candidate_runs()[0]
    assert expected_next(evidence(controls=6, candidates=5, target=True)) == candidate_runs()[5]
    assert expected_next(evidence(controls=6, candidates=6, target=True, analysis=True)) is None


def test_finalist_next_rejects_candidate_or_analysis_before_allowed():
    with pytest.raises(ValueError, match="control-first"):
        expected_next(evidence(controls=1, candidates=1))
    with pytest.raises(ValueError, match="target lock"):
        expected_next(evidence(controls=6))
    with pytest.raises(ValueError, match="before every candidate"):
        expected_next(evidence(controls=6, candidates=1, target=True, analysis=True))


def test_finalist_paths_are_not_created_by_state_checks(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    result = tmp_path / "result.json"
    assert not checkpoint.exists()
    assert not result.exists()

    assert ordered_runs() == control_runs() + candidate_runs()
    assert expected_next(evidence()) == control_runs()[0]

    assert not checkpoint.exists()
    assert not result.exists()


def test_frozen_finalist_runner_matches_automation_contract():
    contract = continuation.load_object(continuation.AUTOMATION)
    assert (
        continuation.file_sha256(continuation.__file__)
        == contract["implementation"]["runner_sha256"]
    )
    assert contract["execution_order"] == ordered_runs()


def test_finalist_transition_audit_preserves_active_v2_boundary():
    root = Path(__file__).parents[1]
    audit = json.loads(
        (
            root / "results" / "Speck-Paper1" / "finalist-automation-transition-audit-v1.json"
        ).read_text(encoding="utf-8")
    )
    assert audit["status"] == (
        "all_12_transitions_simulated_current_v2_preserved_future_hardening_identified"
    )
    assert audit["frozen_inputs"]["runner"]["sha256"] == continuation.file_sha256(
        continuation.__file__
    )
    assert audit["frozen_inputs"]["automation"]["sha256"] == continuation.file_sha256(
        continuation.AUTOMATION
    )
    assert audit["simulation"]["runs"] == 12
    assert audit["simulation"]["commits"] == 12
    assert audit["simulation"]["successor_schedules"] == 11
    assert audit["simulation"]["target_locks"] == 1
    assert audit["simulation"]["final_analyses"] == 1
    assert audit["simulation"]["transition_flags"] == {
        "trigger_disabled_before_collection": True,
        "quality_dependent_branching": False,
        "polling": False,
        "automatic_retry": False,
        "automation_hash_bound": True,
    }
    assert {entry["id"] for entry in audit["residual_hardening"]} == {
        "terminal_service_provenance",
        "finalize_next_run_recheck",
        "systemd_integration_fixture",
    }
    assert audit["decision"]["current_runner_or_contract_modified"] is False
    assert audit["decision"]["current_run_interruption_authorized"] is False
    assert audit["decision"]["future_hardening_applied_mid_sequence"] is False


def test_finalist_finalize_simulates_every_frozen_transition(tmp_path, monkeypatch):
    ordered = ordered_runs()
    program_path = tmp_path / "experiment_program.json"
    transitions = tmp_path / "transitions"
    target_path = tmp_path / "target.json"
    analysis_path = tmp_path / "analysis.json"
    result_dir = tmp_path / "results"
    lock_path = tmp_path / "finalist.lock"
    initial_failed_attempts = [
        {
            "attempt": 1,
            "path": "failed.json",
            "sha256": "f" * 64,
            "status": "operator_interrupted_after_step_1_no_checkpoint_or_result",
        }
    ]
    initial_rerun = {
        "attempt": 2,
        "path": "rerun.json",
        "sha256": "e" * 64,
        "status": "frozen_after_operator_interruption_before_identical_restart",
    }
    program = {
        "finalist_evidence": {
            **evidence(),
            "status": "qualified_unexecuted",
            "next_run": ordered[0],
            "failed_attempts": initial_failed_attempts,
            "active_rerun": initial_rerun,
        }
    }
    continuation.atomic_json(program_path, program)

    events = []
    commits = []
    schedules = []

    def fake_collect(run_name, _entry):
        events.append(("collect", run_name))
        pair = int(run_name.split("-pair-")[1].split("-")[0])
        path = result_dir / f"{run_name}.json"
        continuation.atomic_json(
            path,
            {
                "format": "speck_paper_finalist_run_result",
                "format_version": 2,
                "status": "complete_qualified",
                "pair": {"pair": pair},
                "training_tokens": 1_539_833_856,
                "non_finite_steps": 0,
            },
        )
        return path

    def fake_lock_target(control_entries):
        events.append(("lock_target", len(control_entries)))
        assert [entry["pair"] for entry in control_entries] == list(range(6))
        continuation.atomic_json(
            target_path,
            {"status": "locked_from_six_controls_before_candidates"},
        )
        return {
            "path": target_path.relative_to(tmp_path).as_posix(),
            "sha256": continuation.file_sha256(target_path),
            "status": "locked_from_six_controls_before_candidates",
        }

    def fake_analyze(control_entries, candidate_entries, target_reference):
        events.append(("analyze", len(control_entries), len(candidate_entries)))
        assert [entry["pair"] for entry in control_entries] == list(range(6))
        assert [entry["pair"] for entry in candidate_entries] == list(range(6))
        assert target_reference["status"] == "locked_from_six_controls_before_candidates"
        continuation.atomic_json(
            analysis_path,
            {
                "status": (
                    "complete_crossed_factor_finalist_language_evidence_no_standalone_promotion"
                )
            },
        )
        return {
            "path": analysis_path.relative_to(tmp_path).as_posix(),
            "sha256": continuation.file_sha256(analysis_path),
            "status": (
                "complete_crossed_factor_finalist_language_evidence_no_standalone_promotion"
            ),
        }

    monkeypatch.setattr(continuation, "ROOT", tmp_path)
    monkeypatch.setattr(continuation, "PROGRAM", program_path)
    monkeypatch.setattr(continuation, "RESULTS", result_dir)
    monkeypatch.setattr(continuation, "TRANSITIONS", transitions)
    monkeypatch.setattr(continuation, "TARGET", target_path)
    monkeypatch.setattr(continuation, "ANALYSIS", analysis_path)
    monkeypatch.setattr(continuation, "LOCK", lock_path)
    monkeypatch.setattr(
        continuation,
        "_stop_trigger",
        lambda unit: events.append(("stop_trigger", unit)),
    )
    monkeypatch.setattr(
        continuation,
        "_wait_for_process_exit",
        lambda unit: events.append(("wait_for_exit", unit)),
    )
    monkeypatch.setattr(continuation, "validate_automation_contract", lambda: {})
    monkeypatch.setattr(continuation, "git_clean", lambda: None)
    monkeypatch.setattr(continuation, "_collect", fake_collect)
    monkeypatch.setattr(continuation, "_lock_target", fake_lock_target)
    monkeypatch.setattr(continuation, "_analyze", fake_analyze)
    monkeypatch.setattr(
        continuation,
        "_cpu_command",
        lambda *args: events.append(("validate", *args)),
    )
    monkeypatch.setattr(
        continuation,
        "_commit",
        lambda paths, message: commits.append((list(paths), message)),
    )
    monkeypatch.setattr(
        continuation,
        "_schedule",
        lambda run_name: schedules.append(run_name),
    )

    for index, run_name in enumerate(ordered):
        continuation.finalize(
            run_name,
            f"training-{index}.service",
            f"trigger-{index}.path",
        )
        state = continuation.load_object(program_path)["finalist_evidence"]
        assert state["failed_attempts"] == initial_failed_attempts
        assert state["active_rerun"] == initial_rerun
        if index < len(ordered) - 1:
            assert state["status"] == "in_progress"
            assert state["next_run"] == ordered[index + 1]
        else:
            assert state["status"] == "complete"
            assert state["next_run"] is None

    final = continuation.load_object(program_path)["finalist_evidence"]
    assert [entry["pair"] for entry in final["control_results"]] == list(range(6))
    assert [entry["pair"] for entry in final["candidate_results"]] == list(range(6))
    assert final["time_to_quality_target"]["status"] == (
        "locked_from_six_controls_before_candidates"
    )
    assert final["analysis_result"]["status"] == (
        "complete_crossed_factor_finalist_language_evidence_no_standalone_promotion"
    )
    assert schedules == ordered[1:]
    assert len(commits) == 12
    assert commits[5][1] == "Lock Paper 1 finalist control target"
    assert commits[-1][1] == "Complete Paper 1 finalist paired analysis"
    assert sum(event[0] == "lock_target" for event in events) == 1
    assert sum(event[0] == "analyze" for event in events) == 1
    assert sum(event[0] == "validate" for event in events) == 12
    assert sum(event[0] == "stop_trigger" for event in events) == 12
    assert sum(event[0] == "wait_for_exit" for event in events) == 12
    lock_index = next(index for index, event in enumerate(events) if event[0] == "lock_target")
    first_candidate_index = next(
        index for index, event in enumerate(events) if event == ("collect", candidate_runs()[0])
    )
    last_control_index = next(
        index for index, event in enumerate(events) if event == ("collect", control_runs()[-1])
    )
    analyze_index = next(index for index, event in enumerate(events) if event[0] == "analyze")
    last_candidate_index = next(
        index for index, event in enumerate(events) if event == ("collect", candidate_runs()[-1])
    )
    assert last_control_index < lock_index < first_candidate_index
    assert last_candidate_index < analyze_index

    for index, run_name in enumerate(ordered):
        transition = continuation.load_object(transitions / f"{run_name}.json")
        assert transition["completed_run"] == run_name
        assert transition["next_run"] == (ordered[index + 1] if index < 11 else None)
        assert transition["trigger_disabled_before_collection"] is True
        assert transition["quality_dependent_branching"] is False
        assert transition["polling"] is False
        assert transition["automatic_retry"] is False
        assert transition["automation_contract_sha256"] == continuation.file_sha256(
            continuation.AUTOMATION
        )
