"""Clean-room reference utilities for physical KV-head cache budgets."""

import math


def _salience_matrix(salience):
    if not isinstance(salience, (list, tuple)) or not salience:
        raise ValueError("salience must contain at least one physical cache head")
    rows = []
    width = None
    for row in salience:
        if not isinstance(row, (list, tuple)) or not row:
            raise ValueError("every salience head must contain at least one token")
        values = []
        for value in row:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError("salience values must be finite non-negative numbers")
            values.append(float(value))
        if width is None:
            width = len(values)
        elif len(values) != width:
            raise ValueError("salience heads must have equal token counts")
        rows.append(values)
    return rows


def _budget(budget, heads, tokens):
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise ValueError("cache budget must be an integer")
    if not 0 <= budget <= heads * tokens:
        raise ValueError("cache budget is outside the physical head-token capacity")
    return budget


def top_token_indices(values, budget):
    """Return deterministic within-head top-token identities."""

    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("head salience must contain at least one token")
    matrix = _salience_matrix([values])
    _budget(budget, 1, len(matrix[0]))
    return tuple(sorted(range(len(matrix[0])), key=lambda token: (-matrix[0][token], token))[:budget])


def retained_mass(salience, head_budgets):
    """Sum the salience retained by deterministic per-head top budgets."""

    matrix = _salience_matrix(salience)
    if not isinstance(head_budgets, (list, tuple)) or len(head_budgets) != len(matrix):
        raise ValueError("head budgets must name every physical cache head")
    total = 0.0
    for row, budget in zip(matrix, head_budgets):
        _budget(budget, 1, len(row))
        total += sum(row[token] for token in top_token_indices(row, budget))
    return total


def adaptive_head_budgets(salience, budget):
    """Allocate one layer's total budget by deterministic global top salience."""

    matrix = _salience_matrix(salience)
    heads, tokens = len(matrix), len(matrix[0])
    _budget(budget, heads, tokens)
    ordered = sorted(
        (
            (matrix[head][token], head, token)
            for head in range(heads)
            for token in range(tokens)
        ),
        key=lambda entry: (-entry[0], entry[1], entry[2]),
    )
    selected = ordered[:budget]
    budgets = [0] * heads
    identities = [[] for _ in range(heads)]
    for _, head, token in selected:
        budgets[head] += 1
        identities[head].append(token)
    expected = [list(top_token_indices(row, head_budget)) for row, head_budget in zip(matrix, budgets)]
    if identities != expected:
        raise RuntimeError("global top selection violated per-head prefix identity")
    return {
        "head_budgets": tuple(budgets),
        "selected": tuple((head, token) for _, head, token in selected),
        "selected_by_head": tuple(tuple(values) for values in identities),
        "retained_mass": sum(value for value, _, _ in selected),
    }


def uniform_head_budgets(heads, tokens, budget):
    """Allocate a deterministic quotient/remainder uniform control."""

    if isinstance(heads, bool) or not isinstance(heads, int) or heads < 1:
        raise ValueError("head count must be a positive integer")
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
        raise ValueError("token count must be a positive integer")
    _budget(budget, heads, tokens)
    quotient, remainder = divmod(budget, heads)
    return tuple(quotient + (head < remainder) for head in range(heads))


def eviction_loss_bound(salience, head_budgets, maximum_transformed_value_norm):
    """Evaluate the Ada-KV L1 attention-output upper bound for normalized heads."""

    matrix = _salience_matrix(salience)
    if (
        isinstance(maximum_transformed_value_norm, bool)
        or not isinstance(maximum_transformed_value_norm, (int, float))
        or not math.isfinite(maximum_transformed_value_norm)
        or maximum_transformed_value_norm < 0
    ):
        raise ValueError("maximum transformed-value norm must be finite and non-negative")
    if any(not math.isclose(sum(row), 1.0, rel_tol=0, abs_tol=1e-12) for row in matrix):
        raise ValueError("eviction bound requires every physical head to sum to one")
    kept = retained_mass(matrix, head_budgets)
    bound = 2 * len(matrix) * maximum_transformed_value_norm
    bound -= 2 * maximum_transformed_value_norm * kept
    return max(0.0, bound)
