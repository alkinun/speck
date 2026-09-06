import math

import pytest
import torch
import torch.nn.functional as F

from speck.stable_latentmoe_reference import (
    exact_quantile_bias,
    histogram_quantile_bias,
    histogram_quantile_counts,
    normalized_latent_moe,
    qb_required_bias,
    quantile_routes,
    rms_norm,
    situ_glu,
)


def test_situ_glu_matches_equation_bound_limit_and_gradients():
    gate = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0], dtype=torch.float64, requires_grad=True)
    up = torch.tensor([100.0, -2.0, 0.0, 2.0, 100.0], dtype=torch.float64, requires_grad=True)

    result = situ_glu(gate, up)
    manual = 4 * torch.tanh(gate / 4) * torch.sigmoid(gate)
    manual = manual * 25 * torch.tanh(up / 25)
    assert torch.equal(result, manual)
    assert result.abs().max().item() <= 100
    result.sum().backward()
    assert torch.isfinite(gate.grad).all()
    assert torch.isfinite(up.grad).all()

    small_gate = torch.tensor([-1e-5, 2e-5], dtype=torch.float64)
    small_up = torch.tensor([3e-5, -4e-5], dtype=torch.float64)
    swiglu = F.silu(small_gate) * small_up
    assert torch.allclose(situ_glu(small_gate, small_up), swiglu, rtol=1e-9, atol=1e-20)
    assert torch.allclose(
        situ_glu(gate.detach(), up.detach(), 1e8, 1e8),
        F.silu(gate.detach()) * up.detach(),
        rtol=1e-10,
        atol=1e-10,
    )


def test_rms_norm_matches_explicit_formula_and_positive_scale_invariance():
    value = torch.tensor([[3.0, 4.0], [-2.0, 1.0]], dtype=torch.float64)
    weight = torch.tensor([0.5, 2.0], dtype=torch.float64)
    expected = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + 1e-12) * weight

    assert torch.allclose(rms_norm(value, weight, 1e-12), expected, rtol=0, atol=1e-15)
    assert torch.allclose(
        rms_norm(value * 7, weight, 1e-12), expected, rtol=0, atol=2e-12
    )


def test_normalized_latent_moe_matches_manual_equation_and_has_finite_gradients():
    x = torch.tensor([[1.0, -2.0, 0.5], [-1.0, 0.25, 2.0]], dtype=torch.float64, requires_grad=True)
    down = torch.tensor([[0.5, 1.0, -0.5], [1.5, -0.25, 0.75]], dtype=torch.float64, requires_grad=True)
    up = torch.tensor(
        [[1.0, -0.5], [0.25, 2.0], [-1.0, 0.75]], dtype=torch.float64, requires_grad=True
    )
    routed_matrices = [
        torch.tensor([[1.0, 0.0], [0.5, 1.0]], dtype=torch.float64, requires_grad=True),
        torch.tensor([[0.0, 2.0], [-1.0, 0.5]], dtype=torch.float64, requires_grad=True),
        torch.tensor([[1.0, -1.0], [2.0, 0.0]], dtype=torch.float64, requires_grad=True),
    ]
    shared_matrices = [
        torch.eye(3, dtype=torch.float64, requires_grad=True),
        torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=torch.float64,
            requires_grad=True,
        ),
    ]
    norm_weight = torch.tensor([0.75, 1.25], dtype=torch.float64, requires_grad=True)
    indices = torch.tensor([[0, 2], [1, 0]])
    weights = torch.tensor([[0.25, 0.75], [0.6, 0.4]], dtype=torch.float64)
    routed = [lambda value, matrix=matrix: F.linear(value, matrix) for matrix in routed_matrices]
    shared = [lambda value, matrix=matrix: F.linear(value, matrix) for matrix in shared_matrices]

    result = normalized_latent_moe(
        x,
        down,
        up,
        routed,
        shared,
        indices,
        weights,
        norm_weight=norm_weight,
        eps=1e-12,
    )
    latent = F.linear(x, down)
    all_routed = torch.stack([F.linear(latent, matrix) for matrix in routed_matrices], dim=1)
    selected = torch.gather(all_routed, 1, indices.unsqueeze(-1).expand(-1, -1, 2))
    aggregate = (selected * weights.unsqueeze(-1)).sum(dim=1)
    manual_routed = F.linear(rms_norm(aggregate, norm_weight, 1e-12), up)
    manual_shared = sum(F.linear(x, matrix) for matrix in shared_matrices)

    assert torch.allclose(result["latent_input"], latent, rtol=0, atol=0)
    assert torch.allclose(result["routed_aggregate"], aggregate, rtol=0, atol=0)
    assert torch.allclose(result["output"], manual_shared + manual_routed, rtol=0, atol=0)
    result["output"].square().sum().backward()
    parameters = [x, down, up, norm_weight, *routed_matrices, *shared_matrices]
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in parameters)


def test_normalized_latent_moe_rejects_non_normalized_or_duplicate_routes():
    arguments = (
        torch.ones((1, 2), dtype=torch.float64),
        torch.eye(2, dtype=torch.float64),
        torch.eye(2, dtype=torch.float64),
        [lambda value: value, lambda value: value * 2],
        [lambda value: value],
    )
    with pytest.raises(ValueError, match="sum to one"):
        normalized_latent_moe(
            *arguments,
            torch.tensor([[0, 1]]),
            torch.tensor([[0.4, 0.4]], dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="more than once"):
        normalized_latent_moe(
            *arguments,
            torch.tensor([[0, 0]]),
            torch.tensor([[0.5, 0.5]], dtype=torch.float64),
        )


def test_quantile_routes_use_bias_only_for_selection_and_raw_scores_for_weights():
    scores = torch.tensor([[0.9, 0.8, 0.1], [0.4, 0.3, 0.2]], dtype=torch.float64)
    bias = torch.tensor([-1.0, 0.0, 0.5], dtype=torch.float64)

    routes = quantile_routes(scores, bias, 2)

    assert routes["indices"].tolist() == [[1, 2], [2, 1]]
    assert torch.allclose(routes["weights"][0], torch.tensor([8 / 9, 1 / 9], dtype=torch.float64))
    assert torch.allclose(routes["weights"][1], torch.tensor([0.4, 0.6], dtype=torch.float64))
    assert routes["cutoff_indices"].tolist() == [0, 0]
    assert torch.allclose(
        routes["cutoffs"], torch.tensor([-0.1, -0.6], dtype=torch.float64), rtol=0, atol=1e-16
    )


def test_quantile_routes_break_exact_ties_by_lower_expert_index():
    routes = quantile_routes(
        torch.full((2, 4), 0.5, dtype=torch.float64),
        torch.zeros(4, dtype=torch.float64),
        2,
    )

    assert routes["indices"].tolist() == [[0, 1], [0, 1]]
    assert routes["cutoff_indices"].tolist() == [2, 2]


def test_exact_QB_uses_q_plus_one_order_statistic_and_mean_centers():
    scores = torch.tensor(
        [
            [0.91, 0.20, 0.30, 0.40],
            [0.10, 0.82, 0.31, 0.41],
            [0.11, 0.21, 0.73, 0.42],
            [0.12, 0.22, 0.32, 0.64],
            [0.55, 0.23, 0.33, 0.43],
            [0.13, 0.56, 0.34, 0.44],
            [0.14, 0.24, 0.57, 0.45],
            [0.15, 0.25, 0.35, 0.58],
        ],
        dtype=torch.float64,
    )
    old_bias = torch.tensor([0.03, -0.01, 0.02, -0.04], dtype=torch.float64)

    result = exact_quantile_bias(scores, old_bias, 1)
    sorted_required = torch.sort(qb_required_bias(scores, old_bias, 1), dim=0).values

    assert result["target_load"] == 2
    assert torch.equal(result["uncentered_bias"], sorted_required[2])
    assert result["bias"].mean().item() == pytest.approx(0.0, abs=1e-16)
    assert torch.equal(
        (result["required_bias"] < result["uncentered_bias"]).sum(dim=0),
        torch.full((4,), 2),
    )


def test_exact_QB_is_invariant_to_common_old_bias_offset():
    generator = torch.Generator().manual_seed(7)
    scores = torch.rand((12, 4), generator=generator, dtype=torch.float64)
    bias = torch.tensor([-0.2, 0.1, 0.05, 0.05], dtype=torch.float64)

    base = exact_quantile_bias(scores, bias, 1)
    shifted = exact_quantile_bias(scores, bias + 3.25, 1)

    assert torch.allclose(base["bias"], shifted["bias"], rtol=0, atol=5e-16)
    assert torch.allclose(
        base["uncentered_bias"] + 3.25, shifted["uncentered_bias"], rtol=0, atol=5e-16
    )


def test_histogram_counts_are_exactly_partition_invariant():
    generator = torch.Generator().manual_seed(19)
    scores = torch.rand((80, 4), generator=generator, dtype=torch.float64)
    bias = torch.tensor([-0.1, 0.0, 0.04, 0.06], dtype=torch.float64)
    required = qb_required_bias(scores, bias, 1)

    pooled = histogram_quantile_counts(required, bias, bins=1000)
    shards = [histogram_quantile_counts(part, bias, bins=1000) for part in required.split([7, 31, 42])]
    summed = sum((shard["counts"] for shard in shards), torch.zeros_like(pooled["counts"]))

    assert torch.equal(pooled["counts"], summed)
    assert all(shard["lower"] == pooled["lower"] for shard in shards)
    assert all(shard["upper"] == pooled["upper"] for shard in shards)
    assert pooled["counts"].sum(dim=1).tolist() == [80] * 4


def test_histogram_recovery_follows_appendix_formula_and_quantile_interval():
    required = torch.tensor(
        [
            [-0.91, -0.83],
            [-0.74, -0.62],
            [0.52, 0.41],
            [0.88, 0.79],
        ],
        dtype=torch.float64,
    )
    bias = torch.zeros(2, dtype=torch.float64)
    histogram = histogram_quantile_counts(required, bias, bins=20)
    result = histogram_quantile_bias(histogram["counts"], bias, 1)
    sorted_required = torch.sort(required, dim=0).values
    lower_endpoint = sorted_required[1]
    upper_endpoint = sorted_required[2]

    assert result["target_load"] == 2
    assert result["bin_width"].item() == pytest.approx(0.1)
    assert torch.all(result["uncentered_bias"] >= lower_endpoint)
    assert torch.all(result["uncentered_bias"] <= upper_endpoint)
    assert result["bias"].mean().item() == pytest.approx(0.0, abs=1e-16)


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (lambda: situ_glu(torch.ones(2), torch.ones(3)), "identical shapes"),
        (
            lambda: quantile_routes(torch.tensor([[math.nan, 0.5]]), torch.zeros(2), 1),
            "finite",
        ),
        (
            lambda: exact_quantile_bias(torch.full((3, 4), 0.5), torch.zeros(4), 1),
            "integral target load",
        ),
        (
            lambda: histogram_quantile_bias(torch.tensor([[1, 2], [1, 1]]), torch.zeros(2), 1),
            "same non-zero token count",
        ),
    ],
)
def test_references_reject_invalid_inputs(operation, message):
    with pytest.raises(ValueError, match=message):
        operation()
