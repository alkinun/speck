"""The plan check must reject drift between plan.json, the ladder configurations and the tables."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_program_plan", ROOT / "experiments/main-data/check_program_plan.py"
)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


@pytest.fixture
def plan():
    return json.loads((ROOT / "experiments/main-data/plan.json").read_text())


def validate(tmp_path, plan):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return checker.validate(path)


def test_the_committed_plan_is_consistent():
    result = checker.validate()
    assert result["total_gpu_hours"] == 5000
    assert result["training_authority"] is False


def test_an_unbalanced_budget_is_rejected(tmp_path, plan):
    plan["compute"]["budget_gpu_hours"]["pretraining_ladder"] += 10
    with pytest.raises(ValueError, match="do not sum"):
        validate(tmp_path, plan)


def test_moving_a_line_without_the_program_table_is_rejected(tmp_path, plan):
    budget = plan["compute"]["budget_gpu_hours"]
    budget["pretraining_ladder"] += 100
    budget["reserve"] -= 100
    with pytest.raises(ValueError, match="speck.operations.slurm"):
        validate(tmp_path, plan)
    budget["reserve"] += 100
    budget["evaluation"] -= 100
    with pytest.raises(ValueError, match="program budget row 'Pretraining ladder'"):
        validate(tmp_path, plan)


def test_more_grant_runs_than_the_ladder_budget_affords_are_rejected(tmp_path, plan):
    plan["ladder"]["rungs"][2]["grant_runs"] = 40
    with pytest.raises(ValueError, match="exceed the pretraining ladder budget"):
        validate(tmp_path, plan)


def test_a_rung_size_that_differs_from_its_configuration_is_rejected(tmp_path, plan):
    plan["ladder"]["rungs"][0]["parameters"] += 1
    with pytest.raises(ValueError, match="size differs from its configuration"):
        validate(tmp_path, plan)


def test_a_longer_parent_run_must_fit_its_line_and_table(tmp_path, plan):
    plan["parent"]["target_tokens"] = 100_000_000_000
    with pytest.raises(ValueError, match="exceeds its budget line"):
        validate(tmp_path, plan)
    plan["parent"]["target_tokens"] = 70_000_000_000
    with pytest.raises(ValueError, match="program ladder row 'Parent'"):
        validate(tmp_path, plan)


def test_a_mixture_that_does_not_sum_is_rejected(tmp_path, plan):
    plan["parent"]["starting_mixture"][0]["weight_percent"] += 5
    with pytest.raises(ValueError, match="sum to 100"):
        validate(tmp_path, plan)
