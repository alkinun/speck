import json
from pathlib import Path

from scripts.paper_finalist_systems_pipeline_audit_validate import file_sha256


def test_systems_pipeline_audit_is_bound_and_execution_blocked():
    root = Path(__file__).parents[1]
    path = root / "results" / "Speck-Paper1" / "finalist-systems-pipeline-interface-audit-v1.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    validator = artifact["implementation"]["validator"]
    tests = artifact["implementation"]["tests"]

    assert artifact["status"] == "components_qualified_end_to_end_pipeline_blocked"
    assert file_sha256(root / validator["path"]) == validator["sha256"]
    assert file_sha256(root / tests["path"]) == tests["sha256"]
    assert artifact["activation_boundary"]["blocking_interfaces"] == 7
    assert artifact["decision"]["end_to_end_systems_pipeline_qualified"] is False
    assert artifact["decision"]["systems_execution_authorized"] is False
