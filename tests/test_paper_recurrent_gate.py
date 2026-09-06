import json
from pathlib import Path

import pytest

from scripts.paper_recurrent_gate_validate import validate_design, validate_file

root = Path(__file__).parents[1]
design_path = root / "research" / "paper-1" / "recurrent_gate_readiness_v1.json"
derived = {
    "GDN_differences": [
        (
            f"/blocks/{index}/block/stages/0/branches/0/output_gate_activation",
            "silu",
            "sigmoid",
        )
        for index in (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18)
    ],
    "KDA_activation_field_present": False,
    "KDA_forward_hardcoded_sigmoid": True,
}


def design():
    return json.loads(design_path.read_text(encoding="utf-8"))


def test_recurrent_gate_readiness_reproduces_clean_delta_and_KDA_blocker():
    assert validate_file(design_path) == {
        "status": "valid_training_blocked",
        "clean_GDN_gate_paths": 15,
        "KDA_SiLU_expressible": False,
        "conditional_pairs": 6,
        "training_authorized": False,
    }


def test_recurrent_gate_rejects_false_KDA_configurability_claim():
    value = design()
    value["current_KDA_implementation"]["KDA_SiLU_config_expressible"] = True
    with pytest.raises(ValueError, match="implementation boundary"):
        validate_design(value, root, derived)


def test_recurrent_gate_requires_exact_parent_before_gate_selection():
    value = design()
    value["conditional_experiment"]["parent"] = "generic KDA"
    with pytest.raises(ValueError, match="experiment, analysis, or decision"):
        validate_design(value, root, derived)


def test_recurrent_gate_rejects_general_sigmoid_claim():
    value = design()
    value["decision_rule"]["general_sigmoid_claim"] = True
    with pytest.raises(ValueError, match="experiment, analysis, or decision"):
        validate_design(value, root, derived)


def test_recurrent_gate_rejects_partial_new_run_count():
    value = design()
    value["conditional_experiment"]["new_SiLU_runs_if_sigmoid_reused"] = 3
    with pytest.raises(ValueError, match="experiment, analysis, or decision"):
        validate_design(value, root, derived)


def test_recurrent_gate_rejects_active_code_change_or_training_authority():
    value = design()
    value["registration_boundary"]["active_architecture_or_model_code_changed"] = True
    with pytest.raises(ValueError, match="experiment, analysis, or decision"):
        validate_design(value, root, derived)
