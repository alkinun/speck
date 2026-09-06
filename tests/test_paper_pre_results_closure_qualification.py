import json
from pathlib import Path

from scripts.paper_pre_results_closure_validate import file_sha256


def test_pre_results_closure_qualification_is_bound_and_waiting():
    root = Path(__file__).parents[1]
    path = root / "results" / "Speck-Paper1" / "pre-results-closure-qualified-v1.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    for reference in (
        *artifact["inputs"].values(),
        *artifact["implementation"].values(),
    ):
        assert file_sha256(root / reference["path"]) == reference["sha256"]
    assert artifact["status"] == "qualified_event_wait_required_goal_active_not_blocked"
    assert len(artifact["result"]["closed_autonomous_tracks"]) == 4
    assert artifact["result"]["systems_live_activation_gates"] == 5
    assert artifact["result"]["postcommit_acceptance_v3_external_to_frozen_runner"] is True
    assert artifact["decision"]["event_wait_required"] is True
    assert artifact["decision"]["overall_goal_complete"] is False
    assert artifact["decision"]["blocked_goal_status_appropriate"] is False
    assert artifact["decision"]["new_training_authorized_outside_event_chain"] is False
