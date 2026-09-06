import json
from pathlib import Path

from speck.paper_finalist_systems_telemetry import file_sha256


def test_systems_telemetry_integrator_qualification_is_bound_and_live_sampling_blocked():
    root = Path(__file__).parents[1]
    path = (
        root
        / "results"
        / "Speck-Paper1"
        / "finalist-systems-telemetry-integrator-qualified-v1.json"
    )
    artifact = json.loads(path.read_text(encoding="utf-8"))
    protocol = artifact["inputs"]["protocol"]
    module = artifact["implementation"]["module"]
    cli = artifact["implementation"]["cli"]
    tests = artifact["implementation"]["tests"]

    assert artifact["status"] == (
        "qualified_synthetic_schema_and_energy_integration_live_sampler_blocked"
    )
    assert file_sha256(root / protocol["path"]) == protocol["sha256"]
    assert file_sha256(root / module["path"]) == module["sha256"]
    assert file_sha256(root / cli["path"]) == cli["sha256"]
    assert file_sha256(root / tests["path"]) == tests["sha256"]
    assert artifact["energy_integration"]["missing_lower_power_watts"] == 0
    assert artifact["decision"]["conservative_energy_integrator_qualified"] is True
    assert artifact["decision"]["live_nvidia_smi_or_NVML_sampler_qualified"] is False
    assert artifact["decision"]["execution_authorized"] is False
