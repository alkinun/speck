import json
from pathlib import Path

from scripts.paper_stable_latentmoe_readiness_v2_validate import file_sha256


def test_stable_latentmoe_v2_qualification_is_bound_and_training_blocked():
    root = Path(__file__).parents[1]
    path = root / "results" / "Speck-Paper1" / "stable-latentmoe-readiness-v2-qualified.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    references = (
        artifact["inputs"]["predecessor"],
        artifact["inputs"]["primary_source_audit"],
        artifact["inputs"]["design"],
        artifact["inputs"]["active_experiment_program"],
        artifact["implementation"]["primary_source_validator"],
        artifact["implementation"]["primary_source_tests"],
        artifact["implementation"]["readiness_validator"],
        artifact["implementation"]["readiness_tests"],
    )

    assert artifact["status"] == (
        "qualified_primary_spec_and_CPU_reference_scope_training_parent_hardware_blocked"
    )
    for reference in references:
        assert file_sha256(root / reference["path"]) == reference["sha256"]
    assert artifact["readiness_result"]["six_stage_causal_order_preserved_exactly"] is True
    assert artifact["readiness_result"]["isolated_CPU_primitives_authorized"] == 4
    assert artifact["readiness_result"]["local_model_integration_authorized"] is False
    assert artifact["readiness_result"]["training_authorized"] is False
    assert artifact["decision"]["architecture_choice_made"] is False
