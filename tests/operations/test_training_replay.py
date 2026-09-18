import pytest
import torch

from speck.operations.training_replay import compare_state


def test_complete_optimizer_comparison_rejects_scalar_or_tensor_drift():
    state = {
        "state": {0: {"momentum": torch.tensor([1.0, 2.0]), "step": 4}},
        "param_groups": [{"lr": 0.001, "params": [0]}],
    }
    assert compare_state(state, state, rtol=0, atol=0) == 1
    for other in (
        {**state, "param_groups": [{"lr": 0.002, "params": [0]}]},
        {**state, "state": {0: {"momentum": torch.tensor([1.0, 2.1]), "step": 4}}},
        {**state, "state": {0: {"momentum": torch.tensor([1.0, 2.0]), "step": 5}}},
        {**state, "extra": 1},
    ):
        with pytest.raises(AssertionError):
            compare_state(state, other, rtol=1e-5, atol=1e-6)
