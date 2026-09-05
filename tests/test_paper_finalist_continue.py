from pathlib import Path

import pytest

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


def test_finalist_paths_are_not_created_by_state_checks():
    assert not Path("/mnt/speck-data/speck/paper1-finalist-131m").exists()
    assert not Path("results/Speck-Paper1/finalist-runs").exists()
