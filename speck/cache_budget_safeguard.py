"""Exact integer apportionment for adaptive cache-budget safeguards."""

import math
import re
from fractions import Fraction

from speck.cache_budget import uniform_head_budgets

_RATIONAL = re.compile(r"^(0|[1-9][0-9]*)/([1-9][0-9]*)$")


def exact_uniform_fraction(value):
    """Parse a reproducible exact safeguard fraction in the closed unit interval."""

    if isinstance(value, bool) or isinstance(value, float):
        raise ValueError("uniform fraction must be exact, not boolean or binary float")
    if isinstance(value, int):
        fraction = Fraction(value)
    elif isinstance(value, Fraction):
        fraction = value
    elif isinstance(value, str):
        match = _RATIONAL.fullmatch(value)
        if match is None:
            raise ValueError("uniform fraction string must use numerator/denominator")
        numerator, denominator = map(int, match.groups())
        if math.gcd(numerator, denominator) != 1:
            raise ValueError("uniform fraction string must be reduced")
        fraction = Fraction(numerator, denominator)
    else:
        raise ValueError("uniform fraction must be an integer, Fraction, or rational string")
    if not 0 <= fraction <= 1:
        raise ValueError("uniform fraction must lie in the closed unit interval")
    return fraction


def safeguard_apportionment(adaptive_budgets, token_capacity, uniform_fraction):
    """Mix toward uniform and conserve physical slots by largest remainders."""

    if not isinstance(adaptive_budgets, (list, tuple)) or not adaptive_budgets:
        raise ValueError("adaptive budgets must contain at least one physical cache head")
    if (
        isinstance(token_capacity, bool)
        or not isinstance(token_capacity, int)
        or token_capacity < 1
    ):
        raise ValueError("token capacity must be a positive integer")
    budgets = []
    for budget in adaptive_budgets:
        if isinstance(budget, bool) or not isinstance(budget, int):
            raise ValueError("every adaptive budget must be an integer")
        if not 0 <= budget <= token_capacity:
            raise ValueError("adaptive budget is outside the physical head capacity")
        budgets.append(budget)
    fraction = exact_uniform_fraction(uniform_fraction)
    heads = len(budgets)
    total = sum(budgets)
    uniform = Fraction(total, heads)
    targets = tuple((1 - fraction) * budget + fraction * uniform for budget in budgets)
    floors = tuple(target.numerator // target.denominator for target in targets)
    remaining = total - sum(floors)
    remainders = tuple(target - floor for target, floor in zip(targets, floors))
    remainder_order = tuple(sorted(range(heads), key=lambda head: (-remainders[head], head)))
    apportioned = list(floors)
    for head in remainder_order[:remaining]:
        apportioned[head] += 1
    result = tuple(apportioned)
    if sum(result) != total or any(not 0 <= budget <= token_capacity for budget in result):
        raise RuntimeError("safeguard apportionment violated physical capacity")
    if fraction == 0 and result != tuple(budgets):
        raise RuntimeError("zero uniform fraction did not recover adaptive budgets")
    if fraction == 1 and result != uniform_head_budgets(heads, token_capacity, total):
        raise RuntimeError("full uniform fraction did not recover the uniform control")
    return {
        "head_budgets": result,
        "targets": targets,
        "floors": floors,
        "fractional_remainders": remainders,
        "remainder_order": remainder_order,
        "remaining_slots": remaining,
        "total_budget": total,
        "uniform_fraction": fraction,
    }
