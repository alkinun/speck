import json
from pathlib import Path

from speck.paper_finalist_systems_telemetry import file_sha256


def test_memory_supplement_qualification_is_bound_and_live_support_blocked():
    root = Path(__file__).parents[1]
    path = (
        root / "results" / "Speck-Paper1" / "finalist-systems-memory-supplement-qualified-v1.json"
    )
    artifact = json.loads(path.read_text(encoding="utf-8"))
    supplement = artifact["inputs"]["supplement"]
    module = artifact["implementation"]["module"]
    cli = artifact["implementation"]["cli"]
    tests = artifact["implementation"]["tests"]

    assert artifact["status"] == (
        "qualified_mock_additive_NVML_memory_producer_live_support_blocked"
    )
    assert file_sha256(root / supplement["path"]) == supplement["sha256"]
    assert file_sha256(root / module["path"]) == module["sha256"]
    assert file_sha256(root / cli["path"]) == cli["sha256"]
    assert file_sha256(root / tests["path"]) == tests["sha256"]
    assert artifact["decision"]["static_NVML_memory_producer_gap_closed"] is True
    assert artifact["decision"]["live_memory_used_field_support_qualified"] is False
    assert artifact["decision"]["end_to_end_pipeline_qualified"] is False
    assert artifact["decision"]["systems_execution_authorized"] is False
