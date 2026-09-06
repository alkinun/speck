import json
from pathlib import Path

from scripts.paper_finalist_systems_protocol_validate import file_sha256


def test_finalist_systems_protocol_qualification_is_bound_and_blocked():
    root = Path(__file__).parents[1]
    path = root / "results" / "Speck-Paper1" / "finalist-systems-protocol-qualified-v1.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    protocol = artifact["inputs"]["protocol"]
    validator = artifact["implementation"]["validator"]
    tests = artifact["implementation"]["tests"]

    assert artifact["status"] == (
        "qualified_pre_result_protocol_execution_blocked_on_language_and_implementation_gates"
    )
    assert file_sha256(root / protocol["path"]) == protocol["sha256"]
    assert file_sha256(root / validator["path"]) == validator["sha256"]
    assert file_sha256(root / tests["path"]) == tests["sha256"]
    assert artifact["qualified_design"]["energy_claim_uses_candidate_upper_over_control_lower"]
    assert artifact["qualified_analysis"]["joint_claim_requires_both_primary_endpoints"]
    assert artifact["decision"]["current_execution_authorized"] is False
    assert artifact["decision"]["active_program_changed"] is False
