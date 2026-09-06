import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.kimi_k3_primary_source_validate import (
    validate_audit,
    validate_code_semantics,
    validate_file,
)

root = Path(__file__).parents[1]
audit_path = root / "results" / "Speck-Paper1" / "kimi-k3-primary-source-audit-v1.json"


def audit():
    return json.loads(audit_path.read_text(encoding="utf-8"))


def test_kimi_k3_primary_source_audit_is_valid():
    assert validate_file(audit_path) == {
        "status": "valid",
        "primary_specification_qualified": True,
        "inference_semantics_qualified": True,
        "training_code_qualified": False,
    }


def test_kimi_k3_audit_rejects_training_code_claim():
    value = deepcopy(audit())
    value["source_disposition"]["official_training_implementation_qualified"] = True
    with pytest.raises(ValueError, match="disposition"):
        validate_audit(value, root)


def test_kimi_k3_audit_rejects_QB_code_claim():
    value = deepcopy(audit())
    value["official_code_boundary"]["Quantile_Balancing_update_implemented"] = True
    with pytest.raises(ValueError, match="disposition"):
        validate_audit(value, root)


def test_kimi_k3_semantics_reject_missing_training_error():
    config = {
        "text_config": {
            "hidden_size": 7168,
            "routed_expert_hidden_size": 3584,
            "moe_intermediate_size": 3072,
            "num_experts": 896,
            "num_experts_per_token": 16,
            "num_shared_experts": 2,
            "moe_router_activation_func": "sigmoid",
            "moe_renormalize": True,
            "latent_moe_use_norm": True,
            "activation_situ_beta": 4.0,
            "activation_situ_linear_beta": 25.0,
        }
    }
    with pytest.raises(ValueError, match="code boundary"):
        validate_code_semantics(
            json.dumps(config).encode(),
            b"class SituAndMul: pass",
            (
                b"Permission is hereby granted, free of charge\nModel as a Service\n"
                b"20 million US dollars\n100 million monthly active users"
            ),
        )


def test_kimi_k3_audit_rejects_silent_EMA_requirement():
    value = deepcopy(audit())
    value["official_report_specification"]["histogram_QB"]["EMA"] = "required with decay 0.9"
    with pytest.raises(ValueError, match="disposition"):
        validate_audit(value, root)
