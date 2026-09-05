"""Clean-room GQA reduction utilities for physical KV-head cache budgets."""

import math

from speck.cache_budget import adaptive_head_budgets, retained_mass


def _query_salience_tensor(query_salience):
    if not isinstance(query_salience, (list, tuple)) or not query_salience:
        raise ValueError("query salience must contain at least one physical cache head")
    tensor = []
    query_heads = None
    tokens = None
    for group in query_salience:
        if not isinstance(group, (list, tuple)) or not group:
            raise ValueError("every physical cache head must contain at least one query head")
        if query_heads is None:
            query_heads = len(group)
        elif len(group) != query_heads:
            raise ValueError("GQA groups must contain equal query-head counts")
        rows = []
        for row in group:
            if not isinstance(row, (list, tuple)) or not row:
                raise ValueError("every query head must contain at least one token")
            if tokens is None:
                tokens = len(row)
            elif len(row) != tokens:
                raise ValueError("query heads must contain equal token counts")
            values = []
            for value in row:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError("query salience values must be finite non-negative numbers")
                values.append(float(value))
            rows.append(values)
        tensor.append(rows)
    return tensor


def reduce_grouped_query_salience(query_salience, reduction="mean"):
    """Reduce equal-size query-head groups onto their physical KV heads."""

    tensor = _query_salience_tensor(query_salience)
    if reduction not in {"mean", "sum", "max"}:
        raise ValueError("GQA reduction must be mean, sum, or max")
    reduced = []
    group_size = len(tensor[0])
    tokens = len(tensor[0][0])
    for group in tensor:
        row = []
        for token in range(tokens):
            values = [query_head[token] for query_head in group]
            if reduction == "max":
                row.append(max(values))
            else:
                total = sum(values)
                row.append(total / group_size if reduction == "mean" else total)
        reduced.append(tuple(row))
    return tuple(reduced)


def adaptive_gqa_head_budgets(query_salience, budget, reduction="mean"):
    """Allocate physical slots after an explicit grouped-query reduction."""

    reduced = reduce_grouped_query_salience(query_salience, reduction)
    result = adaptive_head_budgets(reduced, budget)
    return {**result, "reduction": reduction, "reduced_salience": reduced}


def grouped_query_retained_mass(query_salience, head_budgets):
    """Evaluate maximum query-head mass attainable by physical head budgets."""

    summed = reduce_grouped_query_salience(query_salience, "sum")
    return retained_mass(summed, head_budgets)


def grouped_query_selected_mass(query_salience, selected_by_head):
    """Evaluate query-head mass of explicit physical token identities."""

    tensor = _query_salience_tensor(query_salience)
    if not isinstance(selected_by_head, (list, tuple)) or len(selected_by_head) != len(tensor):
        raise ValueError("selected identities must name every physical cache head")
    total = 0.0
    tokens = len(tensor[0][0])
    for group, selected in zip(tensor, selected_by_head):
        if not isinstance(selected, (list, tuple)) or len(set(selected)) != len(selected):
            raise ValueError("selected token identities must be a duplicate-free sequence")
        if any(isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < tokens for token in selected):
            raise ValueError("selected token identity is outside the physical head")
        total += sum(query_head[token] for query_head in group for token in selected)
    return total


def grouped_query_eviction_loss_bound(
    query_salience,
    head_budgets,
    maximum_transformed_value_norm,
):
    """Evaluate the shared-norm query-head bound for equal-size GQA groups."""

    tensor = _query_salience_tensor(query_salience)
    if (
        isinstance(maximum_transformed_value_norm, bool)
        or not isinstance(maximum_transformed_value_norm, (int, float))
        or not math.isfinite(maximum_transformed_value_norm)
        or maximum_transformed_value_norm < 0
    ):
        raise ValueError("maximum transformed-value norm must be finite and non-negative")
    if any(
        not math.isclose(sum(query_head), 1.0, rel_tol=0, abs_tol=1e-12)
        for group in tensor
        for query_head in group
    ):
        raise ValueError("GQA eviction bound requires every query head to sum to one")
    query_heads = len(tensor) * len(tensor[0])
    kept = grouped_query_retained_mass(tensor, head_budgets)
    bound = 2 * query_heads * maximum_transformed_value_norm
    bound -= 2 * maximum_transformed_value_norm * kept
    return max(0.0, bound)
