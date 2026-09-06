import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.paper_sequence_cache_v2_validate import validate_design, validate_file

root = Path(__file__).parents[1]
design_path = root / "research" / "paper-1" / "sequence_cache_representation_v2.json"
geometry = {
    "parameters": 152975898,
    "analytic_flops_per_token_at_4096": 1015703040,
    "attention_branches_changed": 5,
    "ffn_branches_unchanged": 20,
}


def design():
    return json.loads(design_path.read_text(encoding="utf-8"))


def test_cache_representation_v2_recomputes_raw_geometry_and_is_blocked():
    assert validate_file(design_path) == {
        "status": "valid_training_blocked",
        "arms": 4,
        "raw_parameters": 152975898,
        "raw_flops_per_token_at_4096": 1015703040,
        "training_authorized": False,
    }


def test_cache_representation_v2_requires_raw_MQA_arm():
    value = design()
    value["arms"] = [arm for arm in value["arms"] if arm["id"] != "mqa1_raw"]
    with pytest.raises(ValueError, match="arms are incomplete"):
        validate_design(value, root, geometry)


def test_cache_representation_v2_rejects_pure_attribution_to_matched_arm():
    value = design()
    next(arm for arm in value["arms"] if arm["id"] == "mqa1_param_match")[
        "pure_cache_representation_attribution_authorized"
    ] = True
    with pytest.raises(ValueError, match="geometry or attribution"):
        validate_design(value, root, geometry)


def test_cache_representation_v2_rejects_changed_FFN_compensation_contrast():
    value = design()
    value["analysis"]["decomposition_contrast"] = "matched arm only"
    with pytest.raises(ValueError, match="analysis or decision"):
        validate_design(value, root, geometry)


def test_cache_representation_v2_rejects_training_authority():
    value = design()
    value["decision"]["training_authorized"] = True
    with pytest.raises(ValueError, match="analysis or decision"):
        validate_design(value, root, geometry)


def test_cache_representation_v2_preserves_active_program_boundary():
    value = deepcopy(design())
    value["registration_boundary"]["active_experiment_program_changed"] = True
    with pytest.raises(ValueError, match="analysis or decision"):
        validate_design(value, root, geometry)
