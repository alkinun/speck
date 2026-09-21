"""Prose figures must track the numeric plan, in both directions.

The repository's budget revisions have repeatedly left stale GPU-hour figures in Markdown
that `check_program_plan.py` could not see, because it validates arithmetic inside
`plan.json` rather than the documents that restate it. These tests pin the guard that
closes that gap: it must accept the tree as committed, and it must reject a figure that
disagrees with the plan whichever side of the pair moved.
"""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_documents", ROOT / "experiments/main-data/check_documents.py"
)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


@pytest.fixture
def plan():
    return json.loads((ROOT / "experiments/main-data/plan.json").read_text())


def validate_plan(tmp_path, plan):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return checker.validate(path)


def test_committed_documents_agree_with_the_plan():
    result = checker.validate()
    assert result["status"] == "prose_figures_agree_with_numeric_plan"
    assert result["training_admitted"] is False
    # A guard that binds nothing would also pass; require it to be doing real work.
    assert result["bound_figures_checked"] > 40


def test_moving_a_reservation_without_the_documents_is_rejected(tmp_path, plan):
    """The realistic failure: plan.json is revised and the prose is forgotten."""
    reservations = plan["compute"]["reservations_gpu_hours"]
    reservations["capability_and_agentic_mid_training"] += 50
    reservations["protected_recovery_and_evaluation"] -= 50
    with pytest.raises(ValueError, match="mid-training production"):
        validate_plan(tmp_path, plan)


def test_moving_a_research_cap_without_the_documents_is_rejected(tmp_path, plan):
    research = plan["compute"]["data_experiments_breakdown_gpu_hours"]
    research["mid_training"] -= 60
    research["post_training"] += 60
    with pytest.raises(ValueError, match="mid-training research|data research split"):
        validate_plan(tmp_path, plan)


def test_post_training_subdivision_drift_is_rejected(tmp_path, plan):
    """The subdivision that lost its final self-SFT line once already."""
    post = plan["compute"]["proposed_post_training_breakdown_gpu_hours"]
    post["sft"] += 50
    post["final_self_sft"] -= 50
    with pytest.raises(ValueError, match="post-training SFT subdivision"):
        validate_plan(tmp_path, plan)


def test_a_slash_compound_is_checked_as_one_claim(tmp_path, plan):
    """ "700/360/170-hour" is a single claim about three reservations, not three loose numbers."""
    splits = checker._splits(plan)
    assert [description for description, _, _ in splits] == ["data research split"]
    plan["compute"]["data_experiments_breakdown_gpu_hours"]["pretraining"] += 10
    with pytest.raises(ValueError, match="data research split"):
        validate_plan(tmp_path, plan)


def test_the_history_pointer_is_not_governed():
    """Archived documents preserve superseded figures on purpose."""
    assert "archive/" in checker.SKIP
    assert not any("archive/" in str(path) for path in checker._documents())
