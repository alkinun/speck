import json
from pathlib import Path

from speck.paper_finalist_systems_telemetry import file_sha256


def test_runtime_attestation_qualification_is_bound_and_live_probe_blocked():
    root = Path(__file__).parents[1]
    path = (
        root / "results" / "Speck-Paper1" / "finalist-systems-runtime-attestation-qualified-v1.json"
    )
    artifact = json.loads(path.read_text(encoding="utf-8"))
    module = artifact["implementation"]["module"]
    cli = artifact["implementation"]["cli"]
    tests = artifact["implementation"]["tests"]

    assert artifact["status"] == (
        "qualified_mock_identity_and_attestation_builder_live_probe_blocked"
    )
    assert file_sha256(root / module["path"]) == module["sha256"]
    assert file_sha256(root / cli["path"]) == cli["sha256"]
    assert file_sha256(root / tests["path"]) == tests["sha256"]
    assert artifact["runtime_probe_contract"]["graph_breaks"] == 0
    assert artifact["runtime_probe_contract"]["eager_fallbacks"] == 0
    assert artifact["pipeline_effect"]["live_runtime_probe_producer_qualified"] is False
    assert artifact["decision"]["systems_execution_authorized"] is False
