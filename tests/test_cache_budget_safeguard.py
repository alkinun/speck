from fractions import Fraction

import pytest

from speck.cache_budget_safeguard import exact_uniform_fraction, safeguard_apportionment


def test_exact_fraction_parser_rejects_binary_and_noncanonical_inputs():
    assert exact_uniform_fraction("1/5") == Fraction(1, 5)
    assert exact_uniform_fraction(Fraction(1, 2)) == Fraction(1, 2)
    assert exact_uniform_fraction(1) == Fraction(1, 1)
    for value in (0.2, True, "0.2", "2/10", "1/0", "-1/2", "6/5"):
        with pytest.raises(ValueError):
            exact_uniform_fraction(value)


def test_safeguard_conserves_registered_six_slot_witness():
    result = safeguard_apportionment((4, 1, 1), 4, "1/5")

    assert result["targets"] == (Fraction(18, 5), Fraction(6, 5), Fraction(6, 5))
    assert result["floors"] == (3, 1, 1)
    assert result["head_budgets"] == (4, 1, 1)
    assert result["total_budget"] == 6


def test_safeguard_uses_lower_index_for_equal_remainders():
    result = safeguard_apportionment((4, 1, 1), 4, "1/2")

    assert result["targets"] == (Fraction(3), Fraction(3, 2), Fraction(3, 2))
    assert result["remainder_order"] == (1, 2, 0)
    assert result["head_budgets"] == (3, 2, 1)


def test_safeguard_endpoints_recover_adaptive_and_uniform():
    assert safeguard_apportionment((3, 0, 2, 2), 3, "0/1")["head_budgets"] == (
        3,
        0,
        2,
        2,
    )
    assert safeguard_apportionment((3, 0, 2, 2), 3, "1/1")["head_budgets"] == (
        2,
        2,
        2,
        1,
    )


@pytest.mark.parametrize(
    ("adaptive", "capacity", "message"),
    [
        ([], 1, "at least one"),
        ([1], 0, "positive integer"),
        ([True], 1, "must be an integer"),
        ([2], 1, "outside"),
        ([-1], 1, "outside"),
    ],
)
def test_safeguard_rejects_invalid_allocations(adaptive, capacity, message):
    with pytest.raises(ValueError, match=message):
        safeguard_apportionment(adaptive, capacity, "1/2")
