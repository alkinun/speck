import pytest

from speck.cache_budget import (
    adaptive_head_budgets,
    eviction_loss_bound,
    retained_mass,
    top_token_indices,
    uniform_head_budgets,
)


def test_global_top_allocation_returns_physical_head_budgets():
    salience = [[0.7, 0.2, 0.1], [0.4, 0.35, 0.25]]

    result = adaptive_head_budgets(salience, 3)

    assert result["head_budgets"] == (1, 2)
    assert result["selected"] == ((0, 0), (1, 0), (1, 1))
    assert result["selected_by_head"] == ((0,), (0, 1))
    assert result["retained_mass"] == pytest.approx(1.45)


def test_global_ties_use_head_then_token_identity():
    result = adaptive_head_budgets([[0.2] * 5, [0.2] * 5], 7)

    assert result["head_budgets"] == (5, 2)
    assert result["selected"] == (
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 0),
        (1, 1),
    )


def test_uniform_control_conserves_quotient_and_remainder():
    assert uniform_head_budgets(4, 5, 0) == (0, 0, 0, 0)
    assert uniform_head_budgets(4, 5, 7) == (2, 2, 2, 1)
    assert uniform_head_budgets(4, 5, 20) == (5, 5, 5, 5)


def test_retained_mass_uses_deterministic_per_head_prefixes():
    salience = [[0.1, 0.7, 0.2], [0.4, 0.25, 0.35]]

    assert top_token_indices(salience[0], 2) == (1, 2)
    assert retained_mass(salience, (2, 1)) == pytest.approx(1.3)


def test_eviction_bound_is_zero_at_full_budget_and_monotone():
    salience = [[0.7, 0.2, 0.1], [0.4, 0.35, 0.25]]
    bounds = []
    for budget in range(7):
        allocation = adaptive_head_budgets(salience, budget)["head_budgets"]
        bounds.append(eviction_loss_bound(salience, allocation, 3.0))

    assert all(right <= left for left, right in zip(bounds, bounds[1:]))
    assert bounds[-1] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("salience", "budget", "message"),
    [
        ([], 0, "at least one"),
        ([[1.0], [1.0, 0.0]], 1, "equal token counts"),
        ([[float("nan")]], 1, "finite non-negative"),
        ([[1.0]], True, "integer"),
        ([[1.0]], 2, "outside"),
    ],
)
def test_adaptive_allocation_rejects_invalid_inputs(salience, budget, message):
    with pytest.raises(ValueError, match=message):
        adaptive_head_budgets(salience, budget)


def test_eviction_bound_requires_normalized_heads():
    with pytest.raises(ValueError, match="sum to one"):
        eviction_loss_bound([[0.6, 0.6]], [1], 1.0)
