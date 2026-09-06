"""Standalone finite-batch tie policies for the qualified QB CPU oracle."""

import torch

from speck.stable_latentmoe_reference import (
    exact_quantile_bias,
    histogram_quantile_bias,
    histogram_quantile_counts,
    qb_required_bias,
    quantile_routes,
)

POLICIES = (
    "source_upper_endpoint",
    "source_histogram_1000",
    "interval_midpoint",
    "histogram_10000_sensitivity",
)


def coordinate_interval(required_bias, selected_experts):
    """Return the finite-batch load-q threshold interval for every expert."""

    if not isinstance(required_bias, torch.Tensor) or required_bias.ndim != 2:
        raise ValueError("required_bias must be a two-dimensional tensor")
    if required_bias.device.type != "cpu" or not required_bias.is_floating_point():
        raise ValueError("required_bias must be a floating CPU tensor")
    if not torch.isfinite(required_bias).all().item():
        raise ValueError("required_bias must be finite")
    tokens, experts = required_bias.shape
    if (
        isinstance(selected_experts, bool)
        or not isinstance(selected_experts, int)
        or not 0 < selected_experts < experts
    ):
        raise ValueError("selected_experts must be an integer between one and n-1")
    if (tokens * selected_experts) % experts:
        raise ValueError("coordinate interval requires integral target load m*k/n")
    target = tokens * selected_experts // experts
    if not 0 < target < tokens:
        raise ValueError("target load must be strictly between zero and m")
    ordered = torch.sort(required_bias, dim=0, stable=True).values
    return {
        "lower": ordered[target - 1],
        "upper": ordered[target],
        "target_load": target,
    }


def policy_bias(scores, current_bias, selected_experts, policy):
    """Compute one named next-step bias and its coordinate interval."""

    if policy not in POLICIES:
        raise ValueError(f"unknown QB tie policy: {policy}")
    required = qb_required_bias(scores, current_bias, selected_experts)
    interval = coordinate_interval(required, selected_experts)
    if policy == "source_upper_endpoint":
        result = exact_quantile_bias(scores, current_bias, selected_experts)
    elif policy == "interval_midpoint":
        uncentered = (interval["lower"] + interval["upper"]) / 2
        result = {"uncentered_bias": uncentered, "bias": uncentered - uncentered.mean()}
    else:
        bins = 1000 if policy == "source_histogram_1000" else 10000
        histogram = histogram_quantile_counts(required, current_bias, bins)
        result = histogram_quantile_bias(histogram["counts"], current_bias, selected_experts)
    return {
        "bias": result["bias"],
        "uncentered_bias": result["uncentered_bias"],
        "required_bias": required,
        "lower": interval["lower"],
        "upper": interval["upper"],
        "target_load": interval["target_load"],
    }


def route_loads(scores, bias, selected_experts):
    """Return per-expert loads and a boolean assignment matrix."""

    routes = quantile_routes(scores, bias, selected_experts)
    assignment = torch.zeros(scores.shape, dtype=torch.bool)
    assignment.scatter_(1, routes["indices"], True)
    return assignment.sum(dim=0), assignment


def fixed_score_replay(scores, initial_bias, selected_experts, policy, maximum_updates):
    """Replay one QB policy on fixed scores until balance, cycle, or limit."""

    if isinstance(maximum_updates, bool) or not isinstance(maximum_updates, int) or maximum_updates < 1:
        raise ValueError("maximum_updates must be a positive integer")
    bias = initial_bias.clone()
    target = scores.shape[0] * selected_experts // scores.shape[1]
    target_vector = torch.full((scores.shape[1],), target, dtype=torch.int64)
    seen = set()
    previous_assignment = None
    route_churn = []
    coordinate_subgradient_failures = 0
    coordinate_interval_violations = 0
    nonfinite_values = 0
    maximum_absolute_bias = bias.abs().max().item()
    initial_loads = None
    for update in range(maximum_updates + 1):
        loads, assignment = route_loads(scores, bias, selected_experts)
        if initial_loads is None:
            initial_loads = loads.clone()
        if previous_assignment is not None:
            changed_edges = torch.logical_xor(assignment, previous_assignment).sum().item()
            route_churn.append(changed_edges / (2 * scores.shape[0] * selected_experts))
        if torch.equal(loads, target_vector):
            return {
                "status": "perfect_balance",
                "updates": update,
                "initial_loads": initial_loads,
                "final_loads": loads,
                "coordinate_subgradient_failures": coordinate_subgradient_failures,
                "coordinate_interval_violations": coordinate_interval_violations,
                "nonfinite_values": nonfinite_values,
                "maximum_absolute_bias": maximum_absolute_bias,
                "route_churn": route_churn,
            }
        identity = tuple(round(value, 14) for value in bias.tolist())
        if identity in seen:
            return {
                "status": "cycle",
                "updates": update,
                "initial_loads": initial_loads,
                "final_loads": loads,
                "coordinate_subgradient_failures": coordinate_subgradient_failures,
                "coordinate_interval_violations": coordinate_interval_violations,
                "nonfinite_values": nonfinite_values,
                "maximum_absolute_bias": maximum_absolute_bias,
                "route_churn": route_churn,
            }
        if update == maximum_updates:
            return {
                "status": "unresolved",
                "updates": update,
                "initial_loads": initial_loads,
                "final_loads": loads,
                "coordinate_subgradient_failures": coordinate_subgradient_failures,
                "coordinate_interval_violations": coordinate_interval_violations,
                "nonfinite_values": nonfinite_values,
                "maximum_absolute_bias": maximum_absolute_bias,
                "route_churn": route_churn,
            }
        seen.add(identity)
        update_result = policy_bias(scores, bias, selected_experts, policy)
        required = update_result["required_bias"]
        threshold = update_result["uncentered_bias"]
        below = (required < threshold).sum(dim=0)
        below_or_equal = (required <= threshold).sum(dim=0)
        coordinate_subgradient_failures += int(
            ((below > target) | (below_or_equal < target)).sum().item()
        )
        coordinate_interval_violations += int(
            (
                (threshold < update_result["lower"])
                | (threshold > update_result["upper"])
            ).sum().item()
        )
        bias = update_result["bias"]
        nonfinite_values += int((~torch.isfinite(bias)).sum().item())
        maximum_absolute_bias = max(maximum_absolute_bias, bias.abs().max().item())
        previous_assignment = assignment
    raise RuntimeError("unreachable QB replay state")
