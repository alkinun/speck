import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.paper_stable_latentmoe_readiness_v3_validate import validate_design, validate_file

root = Path(__file__).parents[1]
design_path = root / "research" / "paper-1" / "stable_latentmoe_readiness_v3.json"


def design():
    return json.loads(design_path.read_text(encoding="utf-8"))


def test_stable_latentmoe_v3_is_pre_results_complete_but_activation_blocked():
    assert validate_file(design_path) == {
        "status": "valid_pre_results_width_complete_activation_blocked",
        "causal_stages": 6,
        "activation_prerequisites": 7,
        "first_stage_semantics": 11,
        "implementation_authorized": False,
        "training_authorized": False,
    }


def test_stable_latentmoe_v3_rejects_reordered_causal_stage():
    value = design()
    value["required_decomposition_order"].reverse()
    with pytest.raises(ValueError, match="causal order"):
        validate_design(value, root)


def test_stable_latentmoe_v3_rejects_false_expert_parallel_evidence():
    value = design()
    value["new_evidence"]["conventional_MoE_primary_source"][
        "expert_parallel_training_implementation_qualified"
    ] = True
    with pytest.raises(ValueError, match="evidence disposition"):
        validate_design(value, root)


def test_stable_latentmoe_v3_requires_every_activation_prerequisite():
    value = design()
    value["activation_prerequisites"]["sequence_parent_selected"] = True
    with pytest.raises(ValueError, match="activation prerequisite"):
        validate_design(value, root)


def test_stable_latentmoe_v3_rejects_more_pre_results_expansion():
    value = design()
    value["stop_rule"]["no_further_width_source_or_synthetic_expansion_before_activation"] = False
    with pytest.raises(ValueError, match="stop rule"):
        validate_design(value, root)


def test_stable_latentmoe_v3_rejects_implementation_or_training_authority():
    value = deepcopy(design())
    value["decision"]["conventional_MoE_implementation_authorized"] = True
    value["decision"]["training_authorized"] = True
    with pytest.raises(ValueError, match="decision"):
        validate_design(value, root)
