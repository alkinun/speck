from pathlib import Path

import pytest

from scripts.paper_baseline_continue import (
    _candidate_runs,
    _dense_pair_2,
    arguments,
    validate_automation_contract,
)


def test_automatic_sequence_uses_dense_pair_2_then_all_candidates():
    assert "pair-2-seed-44-order-1073741824-dense_global_param_match" in _dense_pair_2()
    assert [f"pair-{pair}" in run for pair, run in enumerate(_candidate_runs())] == [
        True,
        True,
        True,
    ]


def test_checked_automation_contract_matches_runner_and_frozen_inputs():
    contract = validate_automation_contract()

    assert contract["event_contract"]["polling"] is False
    assert contract["decision_contract"]["quality_dependent_branching"] is False


def test_automatic_sequence_arguments_are_explicit():
    finalize = arguments(
        [
            "finalize",
            "run",
            "--training-unit",
            "train.service",
            "--trigger-unit",
            "trigger.path",
        ]
    )
    launch = arguments(["launch", "run"])

    assert finalize.training_unit == "train.service"
    assert finalize.trigger_unit == "trigger.path"
    assert launch.command == "launch"


def test_automatic_sequence_rejects_unknown_launch():
    from scripts.paper_baseline_continue import launch

    with pytest.raises(ValueError, match="candidate"):
        launch("not-a-frozen-run")


def test_cpu_environment_is_separate_from_cuda_environment():
    from scripts.paper_baseline_continue import CPU_ENV, ROOT

    assert CPU_ENV == ROOT / ".venv-paper1-cpu"
    assert CPU_ENV != Path(ROOT / ".venv")
