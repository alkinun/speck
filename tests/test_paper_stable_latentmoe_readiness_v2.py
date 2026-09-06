import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.paper_stable_latentmoe_readiness_v2_validate import validate_design, validate_file

root = Path(__file__).parents[1]
design_path = root / "research" / "paper-1" / "stable_latentmoe_readiness_v2.json"


def design():
    return json.loads(design_path.read_text(encoding="utf-8"))


def test_stable_latentmoe_v2_closes_only_source_gate():
    assert validate_file(design_path) == {
        "status": "valid_CPU_reference_authorized_training_blocked",
        "causal_stages": 6,
        "isolated_CPU_primitives": 4,
        "remaining_hard_blockers": 9,
        "training_authorized": False,
    }


def test_stable_latentmoe_v2_rejects_reordered_causal_stages():
    value = design()
    value["required_decomposition_order"][0], value["required_decomposition_order"][1] = (
        value["required_decomposition_order"][1],
        value["required_decomposition_order"][0],
    )
    with pytest.raises(ValueError, match="causal order"):
        validate_design(value, root)


def test_stable_latentmoe_v2_rejects_training_code_claim():
    value = design()
    value["decision"]["official_training_implementation_qualified"] = True
    with pytest.raises(ValueError, match="disposition"):
        validate_design(value, root)


def test_stable_latentmoe_v2_rejects_silent_EMA():
    value = design()
    value["qualified_source_semantics"]["histogram_Quantile_Balancing"]["EMA_policy"] = (
        "required decay 0.9"
    )
    with pytest.raises(ValueError, match="source semantics"):
        validate_design(value, root)


def test_stable_latentmoe_v2_rejects_removed_hardware_blocker():
    value = design()
    value["remaining_hard_blockers"] = [
        blocker
        for blocker in value["remaining_hard_blockers"]
        if "expert-parallel hardware envelope" not in blocker
    ]
    with pytest.raises(ValueError, match="hard blocker"):
        validate_design(value, root)


def test_stable_latentmoe_v2_rejects_model_integration_authority():
    value = deepcopy(design())
    value["decision"]["local_model_integration_authorized"] = True
    with pytest.raises(ValueError, match="disposition"):
        validate_design(value, root)
