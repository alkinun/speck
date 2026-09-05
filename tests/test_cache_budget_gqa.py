import pytest

from speck.cache_budget import eviction_loss_bound
from speck.cache_budget_gqa import (
    adaptive_gqa_head_budgets,
    grouped_query_eviction_loss_bound,
    grouped_query_retained_mass,
    grouped_query_selected_mass,
    reduce_grouped_query_salience,
)

COUNTEREXAMPLE = [
    [[0.9, 0.1], [0.0, 1.0]],
    [[0.6, 0.4], [0.6, 0.4]],
]


def test_mean_and_sum_reductions_preserve_equal_group_order():
    mean = reduce_grouped_query_salience(COUNTEREXAMPLE, "mean")
    summed = reduce_grouped_query_salience(COUNTEREXAMPLE, "sum")

    assert mean == ((0.45, 0.55), (0.6, 0.4))
    assert summed == ((0.9, 1.1), (1.2, 0.8))
    assert adaptive_gqa_head_budgets(COUNTEREXAMPLE, 1, "mean")["selected"] == ((1, 0),)
    assert adaptive_gqa_head_budgets(COUNTEREXAMPLE, 1, "sum")["selected"] == ((1, 0),)


def test_max_reduction_is_a_strict_negative_control():
    mean = adaptive_gqa_head_budgets(COUNTEREXAMPLE, 1, "mean")
    maximum = adaptive_gqa_head_budgets(COUNTEREXAMPLE, 1, "max")

    assert maximum["selected"] == ((0, 1),)
    assert grouped_query_selected_mass(
        COUNTEREXAMPLE, mean["selected_by_head"]
    ) == pytest.approx(1.2)
    assert grouped_query_retained_mass(COUNTEREXAMPLE, mean["head_budgets"]) == pytest.approx(1.2)
    assert grouped_query_selected_mass(
        COUNTEREXAMPLE, maximum["selected_by_head"]
    ) == pytest.approx(1.1)


def test_query_bound_equals_group_scaled_physical_mean_bound():
    allocation = adaptive_gqa_head_budgets(COUNTEREXAMPLE, 1, "mean")
    direct = grouped_query_eviction_loss_bound(COUNTEREXAMPLE, allocation["head_budgets"], 3.0)
    physical = eviction_loss_bound(
        allocation["reduced_salience"], allocation["head_budgets"], 3.0
    )

    assert direct == pytest.approx(16.8)
    assert direct == pytest.approx(2 * physical)


@pytest.mark.parametrize(
    ("query_salience", "message"),
    [
        ([], "at least one physical"),
        ([[]], "at least one query"),
        ([[[1.0]], [[0.5], [0.5]]], "equal query-head counts"),
        ([[[1.0, 0.0], [1.0]]], "equal token counts"),
        ([[[float("nan")]]], "finite non-negative"),
    ],
)
def test_gqa_reduction_rejects_invalid_shapes_and_values(query_salience, message):
    with pytest.raises(ValueError, match=message):
        reduce_grouped_query_salience(query_salience)


def test_gqa_reduction_rejects_unknown_method():
    with pytest.raises(ValueError, match="mean, sum, or max"):
        reduce_grouped_query_salience([[[1.0]]], "median")


def test_gqa_bound_requires_normalized_query_heads():
    with pytest.raises(ValueError, match="every query head"):
        grouped_query_eviction_loss_bound([[[0.6, 0.6]]], [1], 1.0)


def test_selected_mass_rejects_duplicate_or_out_of_range_identities():
    with pytest.raises(ValueError, match="duplicate-free"):
        grouped_query_selected_mass([[[1.0, 0.0]]], [[0, 0]])
    with pytest.raises(ValueError, match="outside"):
        grouped_query_selected_mass([[[1.0, 0.0]]], [[2]])
