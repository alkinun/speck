import json
from pathlib import Path

from speck.paper import _file_sha256


def test_finalist_program_source_gate_v2_is_bound_to_integration_proof():
    root = Path(__file__).parents[1]
    artifact_path = root / "results" / "Speck-Paper1" / "finalist-program-source-gate-v2.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    predecessor = artifact["predecessor"]
    validator = artifact["implementation"]["program_validator"]
    integration_tests = artifact["implementation"]["integration_tests"]

    assert artifact["format_version"] == 2
    assert artifact["status"] == (
        "complete_finite_11_source_gate_one_control_integration_qualified"
    )
    assert _file_sha256(root / predecessor["path"]) == predecessor["sha256"]
    assert _file_sha256(root / validator["path"]) == validator["sha256"]
    assert _file_sha256(root / integration_tests["path"]) == integration_tests["sha256"]
    assert artifact["integration_fixture"]["complete_control_passes_full_program_validator"]
    assert artifact["integration_fixture"][
        "same_control_with_dclm_removed_rejected_by_full_program_validator"
    ]
    assert artifact["decision"]["v1_mutated_or_replaced"] is False
    assert artifact["decision"]["runner_analyzer_plan_or_threshold_changed"] is False
