import json
from pathlib import Path

from speck.paper_finalist_systems_acceptance import file_sha256


def test_systems_acceptance_qualification_is_bound_and_live_pipeline_blocked():
    root = Path(__file__).parents[1]
    path = root / "results" / "Speck-Paper1" / "finalist-systems-acceptance-qualified-v1.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    module = artifact["implementation"]["module"]
    cli = artifact["implementation"]["cli"]
    tests = artifact["implementation"]["tests"]

    assert artifact["status"] == (
        "qualified_synthetic_global_block_acceptance_live_pipeline_blocked"
    )
    assert file_sha256(root / module["path"]) == module["sha256"]
    assert file_sha256(root / cli["path"]) == cli["sha256"]
    assert file_sha256(root / tests["path"]) == tests["sha256"]
    assert artifact["block_validation"][
        "all_present_complete_blocks_validated_even_if_another_block_failed"
    ]
    assert artifact["analysis_boundary"][
        "frozen_analyzer_called_only_after_six_complete_valid_blocks"
    ]
    assert artifact["pipeline_effect"]["runtime_identity_producer_qualified"] is False
    assert artifact["decision"]["systems_execution_authorized"] is False
