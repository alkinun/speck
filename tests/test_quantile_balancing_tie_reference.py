import pytest
import torch

from speck.quantile_balancing_tie_reference import (
    coordinate_interval,
    fixed_score_replay,
    policy_bias,
)


def test_coordinate_interval_brackets_exact_target_rank():
    required = torch.tensor(
        [[-0.9, -0.7], [-0.3, -0.2], [0.4, 0.5], [0.8, 0.9]], dtype=torch.float64
    )

    interval = coordinate_interval(required, 1)

    assert interval["target_load"] == 2
    assert torch.equal(interval["lower"], torch.tensor([-0.3, -0.2], dtype=torch.float64))
    assert torch.equal(interval["upper"], torch.tensor([0.4, 0.5], dtype=torch.float64))


def test_midpoint_is_inside_coordinate_interval_and_mean_centered():
    scores = torch.tensor(
        [[0.9, 0.2], [0.8, 0.3], [0.4, 0.7], [0.1, 0.6]], dtype=torch.float64
    )
    bias = torch.zeros(2, dtype=torch.float64)

    result = policy_bias(scores, bias, 1, "interval_midpoint")

    assert torch.all(result["uncentered_bias"] >= result["lower"])
    assert torch.all(result["uncentered_bias"] <= result["upper"])
    assert result["bias"].mean().item() == pytest.approx(0, abs=1e-16)


def test_fixed_score_midpoint_replay_reaches_balance_on_discovery_fixture():
    generator = torch.Generator().manual_seed(0)
    scores = torch.rand((8, 4), generator=generator, dtype=torch.float64)
    bias = torch.randn(4, generator=generator, dtype=torch.float64) * 0.15
    bias -= bias.mean()

    result = fixed_score_replay(scores, bias, 1, "interval_midpoint", 50)

    assert result["status"] == "perfect_balance"
    assert torch.equal(result["final_loads"], torch.full((4,), 2))
    assert result["coordinate_subgradient_failures"] == 0
    assert result["coordinate_interval_violations"] == 0


def test_unknown_policy_and_nonintegral_shape_are_rejected():
    scores = torch.full((3, 4), 0.5, dtype=torch.float64)
    bias = torch.zeros(4, dtype=torch.float64)
    with pytest.raises(ValueError, match="unknown QB tie policy"):
        policy_bias(scores, bias, 1, "post_hoc_jitter")
    with pytest.raises(ValueError, match="integral target load"):
        coordinate_interval(scores, 1)
