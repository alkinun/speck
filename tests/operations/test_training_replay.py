from types import SimpleNamespace

import pytest
import torch

from speck.operations.training_replay import compare_state, replay


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


@pytest.mark.parametrize("seconds", [float("nan"), float("inf")])
def test_replay_requires_a_finite_deadline(seconds):
    with pytest.raises(ValueError, match="positive"):
        replay(SimpleNamespace(workers=1, seconds=seconds, checkpoint_step=1))
