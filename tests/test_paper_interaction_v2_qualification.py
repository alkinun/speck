import json
from pathlib import Path

from scripts.paper_interaction_v2_validate import file_sha256


def test_interaction_v2_qualification_is_bound_and_training_blocked():
    root = Path(__file__).parents[1]
    path = root / "results" / "Speck-Paper1" / "interaction-readiness-v2-qualified.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    design = artifact["inputs"]["design"]
    validator = artifact["implementation"]["validator"]
    tests = artifact["implementation"]["tests"]

    assert artifact["status"] == ("qualified_exact_cube_estimands_axes_unselected_training_blocked")
    assert file_sha256(root / design["path"]) == design["sha256"]
    assert file_sha256(root / validator["path"]) == validator["sha256"]
    assert file_sha256(root / tests["path"]) == tests["sha256"]
    assert artifact["qualified_coefficients"]["all_vectors_reconstructed_algorithmically"]
    assert artifact["synthetic_proofs"]["hidden_harm_cube"]["conditional_guard_pass"] is False
    assert artifact["resource_and_authority"]["training_authorized"] is False
    assert artifact["decision"]["v2_registered"] is False
