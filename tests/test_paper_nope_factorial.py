import json
from pathlib import Path

import pytest

from scripts.paper_nope_factorial_validate import validate_design, validate_file

root = Path(__file__).parents[1]
design_path = root / "research" / "paper-1" / "nope_factorial_v1.json"
derived = {
    "GDN_differences": [
        (f"/blocks/{index}/block/stages/0/branches/0/rope_dim", 32, 0)
        for index in (3, 7, 11, 15, 19)
    ],
    "KDA_attention_branches_changed": 5,
    "KDA_RoPE_parameters": 153958938,
    "KDA_RoPE_flops": 1021601280,
}


def design():
    return json.loads(design_path.read_text(encoding="utf-8"))


def test_nope_factorial_recomputes_clean_delta_and_missing_cell_geometry():
    assert validate_file(design_path) == {
        "status": "valid_training_blocked",
        "arms": 4,
        "pairs": 6,
        "reused_future_runs": 6,
        "new_future_runs": 18,
        "training_authorized": False,
    }


def test_nope_factorial_requires_missing_KDA_RoPE_cell():
    value = design()
    value["factorial"]["arms"] = [
        arm for arm in value["factorial"]["arms"] if arm["id"] != "kda_sigmoid_rope32"
    ]
    with pytest.raises(ValueError, match="arm inventory"):
        validate_design(value, root, derived)


def test_nope_factorial_rejects_pooled_or_incomplete_position_contrast():
    value = design()
    value["factorial"]["within_mixer_position_contrasts"] = ["pooled NoPE minus RoPE"]
    with pytest.raises(ValueError, match="analysis, reuse, or decision"):
        validate_design(value, root, derived)


def test_nope_factorial_rejects_outcome_inspection_claim():
    value = design()
    value["reuse_contract"]["outcomes_inspected"] = True
    with pytest.raises(ValueError, match="analysis, reuse, or decision"):
        validate_design(value, root, derived)


def test_nope_factorial_rejects_partial_factorial_run_count():
    value = design()
    value["reuse_contract"]["new_future_runs"] = 12
    with pytest.raises(ValueError, match="analysis, reuse, or decision"):
        validate_design(value, root, derived)


def test_nope_factorial_rejects_training_or_general_claim_authority():
    value = design()
    value["registration_boundary"]["training_authorized"] = True
    with pytest.raises(ValueError, match="analysis, reuse, or decision"):
        validate_design(value, root, derived)
