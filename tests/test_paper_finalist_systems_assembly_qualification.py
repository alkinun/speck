import json
from pathlib import Path

from speck.paper_finalist_systems_assembly import file_sha256


def test_systems_assembly_qualification_is_bound_and_orchestration_blocked():
    root = Path(__file__).parents[1]
    path = root / "results" / "Speck-Paper1" / "finalist-systems-assembly-qualified-v1.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    module = artifact["implementation"]["module"]
    cli = artifact["implementation"]["cli"]
    tests = artifact["implementation"]["tests"]

    assert artifact["status"] == (
        "qualified_synthetic_trial_and_block_assembly_live_orchestration_blocked"
    )
    assert file_sha256(root / module["path"]) == module["sha256"]
    assert file_sha256(root / cli["path"]) == cli["sha256"]
    assert file_sha256(root / tests["path"]) == tests["sha256"]
    assert artifact["paired_batch_consumption"]["missing_fingerprints_cannot_compare_equal"]
    assert artifact["block_assembly"]["complete_block_accepted_by_frozen_analyzer"]
    assert artifact["pipeline_effect"]["runtime_attestation_producer_qualified"] is False
    assert artifact["pipeline_effect"]["end_to_end_pipeline_qualified"] is False
    assert artifact["decision"]["systems_execution_authorized"] is False
