import pytest

from speck.search.planner import (
    ActionProposal,
    commit_plan,
    plan_actions,
    posterior_information,
)
from speck.search.protocol import content_digest
from speck.search.study_v3 import V3Study


def proposals():
    return (
        ActionProposal(
            "continue",
            "quality",
            4.0,
            0.9,
            0.8,
            0.1,
            {"tokens": 100},
        ),
        ActionProposal(
            "profile",
            "efficient",
            2.0,
            0.7,
            0.4,
            0.2,
            {"device": "cpu"},
        ),
        ActionProposal(
            "new_architecture",
            "novel",
            3.0,
            0.2,
            0.7,
            1.0,
            {"operator": "sample"},
        ),
    )


def test_planning_is_deterministic_and_budgeted():
    first = plan_actions(proposals(), available_cost=6.0, max_actions=2, seed=7)
    second = plan_actions(proposals(), available_cost=6.0, max_actions=2, seed=7)
    assert first == second
    assert first.committed_cost <= first.available_cost
    assert len(first.selected) == 2
    assert len(first.digest) == 64


def test_planning_changes_preferences_without_product_thresholds():
    choices = {
        plan_actions(proposals(), 4.0, 1, seed).selected[0].proposal.architecture_digest
        for seed in range(20)
    }
    assert len(choices) > 1


def test_plans_commit_as_replayable_worker_actions(tmp_path):
    study = V3Study(tmp_path / "study.sqlite3")
    study.initialize({}, {})
    decision = plan_actions(proposals(), available_cost=6.0, max_actions=2, seed=9)
    action_ids = commit_plan(study, decision)
    assert len(action_ids) == 2
    assert len(study.actions("pending")) == 2
    assert study.events()[-3]["kind"] == "planning_decision"
    for action in study.actions("pending"):
        assert action["payload"]["planning_decision_digest"] == decision.digest
    event_count = len(study.events())
    assert commit_plan(study, decision) == action_ids
    assert len(study.events()) == event_count
    study.close()


def test_posterior_information_tracks_expected_variance_reduction():
    covariance = ((4.0, 0.0), (0.0, 9.0))
    assert posterior_information(covariance, 0.0) == 0.0
    assert posterior_information(covariance, 0.5) > 0
    assert posterior_information(covariance, 1.0) == (
        2 * posterior_information(covariance, 0.5)
    )


def test_planning_batch_rolls_back_as_one_transaction(tmp_path):
    study = V3Study(tmp_path / "study.sqlite3")
    study.initialize({}, {})
    events = len(study.events())
    definition = {"seed": 1}
    with pytest.raises(ValueError, match="positive"):
        study.commit_planning_decision(
            content_digest(definition),
            definition,
            (
                {
                    "kind": "profile",
                    "priority": 1.0,
                    "estimated_cost": 0.0,
                    "payload": {},
                },
            ),
        )
    assert study.actions() == []
    assert len(study.events()) == events
    study.close()
