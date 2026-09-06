import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.deepseek_moe_primary_source_validate import (
    validate_audit,
    validate_code_semantics,
    validate_file,
)

root = Path(__file__).parents[1]
audit_path = root / "results" / "Speck-Paper1" / "deepseek-moe-primary-source-audit-v1.json"


def audit():
    return json.loads(audit_path.read_text(encoding="utf-8"))


def test_deepseek_moe_primary_source_audit_is_valid_and_implementation_blocked():
    assert validate_file(audit_path) == {
        "status": "valid_implementation_blocked",
        "primary_equations_qualified": True,
        "single_device_dropless_training_semantics": True,
        "expert_parallel_training_qualified": False,
    }


def test_deepseek_moe_audit_rejects_selected_weight_renormalization_claim():
    value = deepcopy(audit())
    value["official_16B_code_boundary"]["selected_weight_renormalization_config"] = True
    with pytest.raises(ValueError, match="disposition"):
        validate_audit(value, root)


def test_deepseek_moe_audit_rejects_expert_parallel_training_claim():
    value = deepcopy(audit())
    value["source_disposition"]["official_expert_parallel_training_implementation_qualified"] = True
    with pytest.raises(ValueError, match="disposition"):
        validate_audit(value, root)


def test_deepseek_moe_audit_rejects_local_implementation_authority():
    value = deepcopy(audit())
    value["source_disposition"]["local_conventional_MoE_implementation_authorized"] = True
    with pytest.raises(ValueError, match="disposition"):
        validate_audit(value, root)


def test_deepseek_moe_semantics_reject_capacity_or_drop_branch():
    config = {
        "hidden_size": 2048,
        "moe_intermediate_size": 1408,
        "n_routed_experts": 64,
        "num_experts_per_tok": 6,
        "n_shared_experts": 2,
        "scoring_func": "softmax",
        "norm_topk_prob": False,
        "first_k_dense_replace": 1,
    }
    with pytest.raises(ValueError, match="semantics changed|dispatch boundary"):
        validate_code_semantics(
            json.dumps(config).encode(),
            b"class MoEGate(nn.Module): capacity_factor = 1",
            b"class Gate(nn.Module): all_to_all()",
            b"MIT License Permission is hereby granted",
        )
