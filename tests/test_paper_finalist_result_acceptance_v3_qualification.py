import json
from pathlib import Path

from scripts.paper_finalist_result_acceptance_validate import file_sha256


def test_finalist_result_acceptance_v3_is_bound_to_git_provenance():
    root = Path(__file__).parents[1]
    path = root / "results" / "Speck-Paper1" / "finalist-result-acceptance-qualified-v3.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    predecessor = artifact["predecessor"]
    verifier = artifact["implementation"]["verifier"]
    tests = artifact["implementation"]["tests"]

    assert artifact["status"] == (
        "qualified_append_only_git_event_successor_before_first_accepted_result"
    )
    assert file_sha256(root / predecessor["path"]) == predecessor["sha256"]
    assert file_sha256(root / verifier["path"]) == verifier["sha256"]
    assert file_sha256(root / tests["path"]) == tests["sha256"]
    assert artifact["event_commit_contract"]["commit_file_set_exact"] is True
    assert artifact["event_commit_contract"]["program_snapshot_must_equal_exact_accepted_prefix"]
    assert artifact["decision"]["v3_supersedes_v2_as_operational_postcommit_command"]
    assert (
        artifact["decision"]["active_runner_collector_analyzer_program_or_training_modified"]
        is False
    )
