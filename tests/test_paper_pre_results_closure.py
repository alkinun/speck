import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.paper_pre_results_closure_validate import validate_closure, validate_file

root = Path(__file__).parents[1]
closure_path = root / "research" / "paper-1" / "pre_results_closure_v1.json"


def closure():
    return json.loads(closure_path.read_text(encoding="utf-8"))


def test_pre_results_closure_requires_event_wait():
    assert validate_file(closure_path) == {
        "status": "valid_event_wait_required",
        "closed_autonomous_tracks": 4,
        "systems_live_gates": 5,
        "external_authority_domains": 2,
        "allowed_actions": 6,
        "goal_complete": False,
    }


def test_pre_results_closure_rejects_false_result_acceptance_wiring():
    value = closure()
    value["operational_chain"]["state"]["postcommit_v3_wired_into_frozen_runner"] = True
    with pytest.raises(ValueError, match="operational state"):
        validate_closure(value, root)


def test_pre_results_closure_rejects_live_systems_authority():
    value = closure()
    value["closed_autonomous_tracks"]["systems"]["systems_execution_authorized"] = True
    with pytest.raises(ValueError, match="track closure"):
        validate_closure(value, root)


def test_pre_results_closure_rejects_local_HELMET_authority():
    value = closure()
    value["external_authority_gates"]["HELMET"]["local_next_action_available"] = True
    with pytest.raises(ValueError, match="HELMET"):
        validate_closure(value, root)


def test_pre_results_closure_rejects_self_review_as_independent():
    value = closure()
    value["external_authority_gates"]["novelty"]["local_next_action_available"] = True
    with pytest.raises(ValueError, match="novelty"):
        validate_closure(value, root)


def test_pre_results_closure_rejects_false_completion_or_training_authority():
    value = deepcopy(closure())
    value["decision"]["overall_goal_complete"] = True
    value["decision"]["new_training_authorized_outside_event_chain"] = True
    with pytest.raises(ValueError, match="decision"):
        validate_closure(value, root)
