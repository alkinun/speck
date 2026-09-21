from types import SimpleNamespace

import pytest
import torch

from speck.operations.training_replay import (
    _train_command,
    compare_metadata,
    compare_state,
    replay,
)


@pytest.mark.parametrize(
    "phase,counters",
    [
        ("base", {"global_step": 4, "global_tokens": 16384}),
        ("sft", {"trained_supervised_tokens": 60413}),
    ],
)
def test_replay_metadata_requires_matching_production_counters(phase, counters):
    expected = {
        "step": 4,
        "training_phase": phase,
        "manifest": "frozen-data",
        "rng_state": {},
        "data_state": {},
        **counters,
    }
    compare_metadata(expected, dict(expected))
    for key, value in counters.items():
        with pytest.raises(AssertionError, match=key):
            compare_metadata(expected, {**expected, key: value + 1})
        missing = {k: v for k, v in expected.items() if k != key}
        with pytest.raises(AssertionError, match=key):
            compare_metadata(expected, missing)
        with pytest.raises(AssertionError, match=key):
            compare_metadata(missing, missing)
    with pytest.raises(AssertionError, match="manifest"):
        compare_metadata(expected, {**expected, "manifest": "other-data"})


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


@pytest.mark.parametrize("compiled", [False, True])
def test_replay_command_controls_compilation(compiled):
    # The selected production recipe is compiled, so the replay must be able to
    # qualify the compiled distributed path; eager stays the default.
    command = _train_command(
        SimpleNamespace(phase="base", device="cuda", compile=compiled), "/tmp/experiment"
    )
    assert ("--no-compile" in command) is not compiled
