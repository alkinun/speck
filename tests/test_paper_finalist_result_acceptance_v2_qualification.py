import json
from pathlib import Path

from scripts.paper_finalist_result_acceptance_validate import file_sha256


def test_finalist_result_acceptance_v2_is_bound_to_checkpoint_replay():
    root = Path(__file__).parents[1]
    path = root / "results" / "Speck-Paper1" / "finalist-result-acceptance-qualified-v2.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    predecessor = artifact["predecessor"]
    verifier = artifact["implementation"]["verifier"]
    tests = artifact["implementation"]["tests"]
    collector = artifact["implementation"]["frozen_collector_module"]

    assert artifact["status"] == (
        "qualified_checkpoint_replay_successor_before_first_accepted_result"
    )
    assert file_sha256(root / predecessor["path"]) == predecessor["sha256"]
    assert file_sha256(root / verifier["path"]) == verifier["sha256"]
    assert file_sha256(root / tests["path"]) == tests["sha256"]
    assert file_sha256(root / collector["path"]) == collector["sha256"]
    assert artifact["replay_contract"]["stable_result_fields_compared_exactly"] is True
    assert artifact["decision"]["must_run_after_every_automatic_result_commit"] is True
    assert artifact["decision"]["active_runner_collector_analyzer_or_program_modified"] is False
