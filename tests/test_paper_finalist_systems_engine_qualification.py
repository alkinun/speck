import json
from pathlib import Path

from speck.paper_finalist_systems_engine import file_sha256


def test_systems_engine_qualification_is_bound_and_runtime_blocked():
    root = Path(__file__).parents[1]
    path = root / "results" / "Speck-Paper1" / "finalist-systems-engine-qualified-v1.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    workload = artifact["inputs"]["workload_plan_qualification"]
    module = artifact["implementation"]["module"]
    cli = artifact["implementation"]["cli"]
    tests = artifact["implementation"]["tests"]

    assert artifact["status"] == (
        "qualified_CPU_fixture_control_flow_GPU_checkpoint_and_execution_blocked"
    )
    assert file_sha256(root / workload["path"]) == workload["sha256"]
    assert file_sha256(root / module["path"]) == module["sha256"]
    assert file_sha256(root / cli["path"]) == cli["sha256"]
    assert file_sha256(root / tests["path"]) == tests["sha256"]
    assert artifact["paired_batch_contract"]["actual_microbatches_per_trial"] == 161
    assert artifact["activation_firewall"]["current_activation_artifact_present"] is False
    assert artifact["decision"]["benchmark_engine_runtime_qualified"] is False
    assert artifact["decision"]["systems_execution_authorized"] is False
