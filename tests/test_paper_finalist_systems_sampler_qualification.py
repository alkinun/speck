import json
from pathlib import Path

from speck.paper_finalist_systems_telemetry import file_sha256


def test_systems_sampler_qualification_is_bound_and_live_support_blocked():
    root = Path(__file__).parents[1]
    path = root / "results" / "Speck-Paper1" / "finalist-systems-sampler-qualified-v1.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    protocol = artifact["inputs"]["protocol"]
    module = artifact["implementation"]["module"]
    cli = artifact["implementation"]["cli"]
    tests = artifact["implementation"]["tests"]

    assert artifact["status"] == "qualified_mock_acquisition_live_field_support_blocked"
    assert file_sha256(root / protocol["path"]) == protocol["sha256"]
    assert file_sha256(root / module["path"]) == module["sha256"]
    assert file_sha256(root / cli["path"]) == cli["sha256"]
    assert file_sha256(root / tests["path"]) == tests["sha256"]
    assert artifact["host_query"]["disk_devices_must_be_explicit"] is True
    assert artifact["host_query"]["automatic_all_device_aggregation"] is False
    assert artifact["decision"]["live_GPU_field_availability_qualified"] is False
    assert artifact["decision"]["execution_authorized"] is False
