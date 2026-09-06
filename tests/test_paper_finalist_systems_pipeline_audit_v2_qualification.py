import json
from pathlib import Path

from scripts.paper_finalist_systems_pipeline_audit_v2_validate import file_sha256


def test_v2_pipeline_audit_is_bound_and_activation_absent():
    root = Path(__file__).parents[1]
    path = root / "results" / "Speck-Paper1" / "finalist-systems-pipeline-interface-audit-v2.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    predecessor = artifact["predecessor"]
    validator = artifact["implementation"]["validator"]
    tests = artifact["implementation"]["tests"]

    assert artifact["status"] == "static_interfaces_composed_live_activation_gates_blocked"
    assert file_sha256(root / predecessor["path"]) == predecessor["sha256"]
    assert file_sha256(root / validator["path"]) == validator["sha256"]
    assert file_sha256(root / tests["path"]) == tests["sha256"]
    assert len(artifact["remaining_activation_gates"]) == 5
    assert artifact["activation"]["artifact_present"] is False
    assert artifact["decision"]["further_live_work_during_language_sequence_authorized"] is False
    assert artifact["decision"]["systems_execution_authorized"] is False
