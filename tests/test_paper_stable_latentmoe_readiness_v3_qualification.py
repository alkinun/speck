import json
from pathlib import Path

from scripts.paper_stable_latentmoe_readiness_v3_validate import file_sha256


def test_stable_latentmoe_v3_qualification_is_bound_and_activation_blocked():
    root = Path(__file__).parents[1]
    path = root / "results" / "Speck-Paper1" / "stable-latentmoe-readiness-v3-qualified.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    for reference in (
        artifact["inputs"]["design"],
        artifact["inputs"]["active_experiment_program"],
        artifact["implementation"]["validator"],
        artifact["implementation"]["tests"],
    ):
        assert file_sha256(root / reference["path"]) == reference["sha256"]
    assert artifact["status"] == "qualified_pre_results_width_complete_activation_blocked"
    assert artifact["convergence"]["activation_prerequisites_open"] == 7
    assert artifact["convergence"]["pre_results_width_source_or_synthetic_expansion_stopped"] is True
    assert artifact["decision"]["pre_results_width_reference_work_complete"] is True
    assert artifact["decision"]["width_architecture_selected"] is False
    assert artifact["decision"]["conventional_MoE_implementation_authorized"] is False
    assert artifact["decision"]["training_authorized"] is False
