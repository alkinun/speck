import json
from pathlib import Path

from speck.paper_finalist_systems_telemetry import file_sha256


def test_orchestration_qualification_is_bound_and_live_adapter_blocked():
    root = Path(__file__).parents[1]
    path = root / "results" / "Speck-Paper1" / "finalist-systems-orchestration-qualified-v1.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    module = artifact["implementation"]["module"]
    tests = artifact["implementation"]["tests"]

    assert artifact["status"] == ("qualified_recording_adapter_state_machine_live_adapter_blocked")
    assert file_sha256(root / module["path"]) == module["sha256"]
    assert file_sha256(root / tests["path"]) == tests["sha256"]
    assert artifact["state_machine_contract"][
        "paired_temperature_difference_checked_before_second_engine"
    ]
    assert artifact["state_machine_contract"]["final_block_successor_authority"] is False
    assert artifact["pipeline_effect"]["live_adapter_qualified"] is False
    assert artifact["decision"]["systems_execution_authorized"] is False
