from copy import deepcopy
from pathlib import Path

import pytest
import torch

from scripts.stable_latentmoe_cpu_reference_qualify import (
    _interval_error,
    load_object,
    validate_protocol,
)

root = Path(__file__).parents[1]
protocol_path = root / "research" / "paper-1" / "stable_latentmoe_cpu_reference_v1.json"


def protocol():
    return load_object(protocol_path)


def test_stable_latentmoe_CPU_protocol_is_bound_and_training_blocked():
    matrix = validate_protocol(protocol(), root)

    assert matrix["dtype"] == "torch.float64"
    assert matrix["Quantile_Balancing"]["required_production_fidelity_bins"] == 1000
    assert protocol()["boundary"]["primitive_composition_authorized"] is False
    assert protocol()["boundary"]["training_authorized"] is False


def test_stable_latentmoe_CPU_protocol_rejects_training_authority():
    value = deepcopy(protocol())
    value["boundary"]["training_authorized"] = True
    with pytest.raises(ValueError, match="contract changed"):
        validate_protocol(value, root)


def test_stable_latentmoe_CPU_protocol_rejects_EMA():
    value = deepcopy(protocol())
    value["primitives"]["histogram_Quantile_Balancing"]["EMA_included"] = True
    with pytest.raises(ValueError, match="contract changed"):
        validate_protocol(value, root)


def test_quantile_interval_error_is_zero_inside_and_distance_outside():
    lower = torch.tensor([-0.5, 0.0], dtype=torch.float64)
    upper = torch.tensor([0.5, 1.0], dtype=torch.float64)

    assert torch.equal(
        _interval_error(torch.tensor([0.25, 0.75]), lower, upper),
        torch.zeros(2, dtype=torch.float64),
    )
    assert torch.equal(
        _interval_error(torch.tensor([-0.75, 1.5]), lower, upper),
        torch.tensor([0.25, 0.5], dtype=torch.float64),
    )
