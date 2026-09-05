import math

import pytest

from scripts.cache_equivalence_power import required_cases, rounded_source_balance


def test_power_requires_more_cases_for_more_variance_or_smaller_margin():
    baseline = required_cases(0.02, 0.01)

    assert required_cases(0.04, 0.01) > baseline
    assert required_cases(0.02, 0.005) > baseline
    assert required_cases(0.0, 0.01) == 1


def test_power_uses_the_frozen_one_sided_90_percent_formula():
    expected = math.ceil(((1.6448536269514722 + 1.2815515655446004) * 0.02 / 0.01) ** 2)

    assert required_cases(0.02, 0.01) == expected
    assert rounded_source_balance(expected) % 11 == 0
    with pytest.raises(ValueError, match="90%"):
        required_cases(0.02, 0.01, power=0.8)
