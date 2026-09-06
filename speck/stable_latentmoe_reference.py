"""Isolated CPU references for the Kimi K3 Stable LatentMoE equations.

These routines favor explicit source semantics and adversarial validation over
performance. They are not wired into Speck's model or training paths.
"""

from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F


def _cpu_tensor(value, name, *, dimensions=None, floating=None):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch tensor")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must remain on CPU in the isolated reference")
    if dimensions is not None and value.ndim != dimensions:
        raise ValueError(f"{name} must have {dimensions} dimensions")
    if floating is True and not value.is_floating_point():
        raise ValueError(f"{name} must use a floating dtype")
    if floating is False and value.is_floating_point():
        raise ValueError(f"{name} must use an integer dtype")
    return value


def _finite(value, name):
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must be finite")


def situ_glu(gate, up, gate_beta=4.0, up_beta=25.0):
    """Apply K3 Eq. 12 to already projected gate and up branches."""

    gate = _cpu_tensor(gate, "gate", floating=True)
    up = _cpu_tensor(up, "up", floating=True)
    if gate.shape != up.shape:
        raise ValueError("gate and up branches must have identical shapes")
    _finite(gate, "gate")
    _finite(up, "up")
    for value, name in ((gate_beta, "gate_beta"), (up_beta, "up_beta")):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{name} must be a positive number")
    gate_cap = gate_beta * torch.tanh(gate / gate_beta)
    up_cap = up_beta * torch.tanh(up / up_beta)
    return gate_cap * torch.sigmoid(gate) * up_cap


def rms_norm(value, weight=None, eps=1e-6):
    """Apply an explicit last-dimension RMSNorm used by K3 Eq. 11."""

    value = _cpu_tensor(value, "value", floating=True)
    if value.ndim < 1 or value.shape[-1] < 1:
        raise ValueError("value must have a non-empty final dimension")
    _finite(value, "value")
    if isinstance(eps, bool) or not isinstance(eps, (int, float)) or eps <= 0:
        raise ValueError("eps must be a positive number")
    if weight is not None:
        weight = _cpu_tensor(weight, "weight", dimensions=1, floating=True)
        if weight.shape[0] != value.shape[-1]:
            raise ValueError("RMSNorm weight must match the final dimension")
        _finite(weight, "weight")
    return F.rms_norm(value, (value.shape[-1],), weight=weight, eps=eps)


def _expert_outputs(value, experts, name):
    if not isinstance(experts, Sequence) or not experts:
        raise ValueError(f"{name} must contain at least one expert")
    outputs = []
    for expert in experts:
        if not isinstance(expert, Callable):
            raise ValueError(f"every {name} entry must be callable")
        output = expert(value)
        output = _cpu_tensor(output, f"{name} output", dimensions=2, floating=True)
        if output.shape != value.shape:
            raise ValueError(f"every {name} output must preserve its input shape")
        _finite(output, f"{name} output")
        outputs.append(output)
    return torch.stack(outputs, dim=1)


def normalized_latent_moe(
    x,
    down_weight,
    up_weight,
    routed_experts,
    shared_experts,
    selected_indices,
    selected_weights,
    *,
    norm_weight=None,
    eps=1e-6,
):
    """Evaluate K3 Eq. 11 densely as a tiny, differentiable CPU oracle."""

    x = _cpu_tensor(x, "x", dimensions=2, floating=True)
    down_weight = _cpu_tensor(down_weight, "down_weight", dimensions=2, floating=True)
    up_weight = _cpu_tensor(up_weight, "up_weight", dimensions=2, floating=True)
    selected_indices = _cpu_tensor(
        selected_indices, "selected_indices", dimensions=2, floating=False
    )
    selected_weights = _cpu_tensor(
        selected_weights, "selected_weights", dimensions=2, floating=True
    )
    for value, name in (
        (x, "x"),
        (down_weight, "down_weight"),
        (up_weight, "up_weight"),
        (selected_weights, "selected_weights"),
    ):
        _finite(value, name)
    tokens, width = x.shape
    latent = down_weight.shape[0]
    if down_weight.shape[1] != width or up_weight.shape != (width, latent):
        raise ValueError("down/up projection shapes must map width to latent and back")
    if selected_indices.shape != selected_weights.shape or selected_indices.shape[0] != tokens:
        raise ValueError("selected indices and weights must name every token and route")
    if selected_indices.dtype != torch.int64:
        raise ValueError("selected routed expert indices must use int64")
    if selected_indices.shape[1] < 1:
        raise ValueError("at least one routed expert must be selected")
    weight_tolerance = 16 * torch.finfo(selected_weights.dtype).eps
    if (selected_weights < 0).any().item() or not torch.allclose(
        selected_weights.sum(dim=-1),
        torch.ones(tokens, dtype=selected_weights.dtype),
        rtol=0,
        atol=weight_tolerance,
    ):
        raise ValueError("selected routed weights must be non-negative and sum to one")
    if any(len(set(row.tolist())) != len(row) for row in selected_indices):
        raise ValueError("a token cannot select one routed expert more than once")

    latent_input = F.linear(x, down_weight)
    routed_all = _expert_outputs(latent_input, routed_experts, "routed_experts")
    if (selected_indices < 0).any().item() or (selected_indices >= routed_all.shape[1]).any().item():
        raise ValueError("selected routed expert index is outside the expert pool")
    gather = selected_indices.unsqueeze(-1).expand(-1, -1, latent)
    selected = torch.gather(routed_all, 1, gather)
    routed_aggregate = (selected * selected_weights.unsqueeze(-1)).sum(dim=1)
    normalized_aggregate = rms_norm(routed_aggregate, norm_weight, eps)
    routed_output = F.linear(normalized_aggregate, up_weight)
    shared_all = _expert_outputs(x, shared_experts, "shared_experts")
    shared_output = shared_all.sum(dim=1)
    return {
        "output": shared_output + routed_output,
        "latent_input": latent_input,
        "routed_aggregate": routed_aggregate,
        "normalized_aggregate": normalized_aggregate,
        "routed_output": routed_output,
        "shared_output": shared_output,
    }


def _routing_inputs(scores, bias, selected_experts):
    scores = _cpu_tensor(scores, "scores", dimensions=2, floating=True)
    bias = _cpu_tensor(bias, "bias", dimensions=1, floating=True)
    _finite(scores, "scores")
    _finite(bias, "bias")
    if scores.shape[1] != bias.shape[0] or scores.shape[0] < 1:
        raise ValueError("scores and bias must name the same non-empty expert pool")
    if (scores < 0).any().item() or (scores > 1).any().item():
        raise ValueError("scores must be sigmoid probabilities in [0, 1]")
    if (
        isinstance(selected_experts, bool)
        or not isinstance(selected_experts, int)
        or not 0 < selected_experts < scores.shape[1]
    ):
        raise ValueError("selected_experts must be an integer between one and n-1")
    return scores, bias


def quantile_routes(scores, bias, selected_experts):
    """Evaluate K3 Eq. 13 plus the biased top-(k+1) cutoff."""

    scores, bias = _routing_inputs(scores, bias, selected_experts)
    biased = scores + bias.unsqueeze(0)
    order = torch.argsort(biased, dim=-1, descending=True, stable=True)
    selected = order[:, :selected_experts]
    selected_raw = torch.gather(scores, 1, selected)
    denominator = selected_raw.sum(dim=-1, keepdim=True)
    if (denominator <= 0).any().item():
        raise ValueError("selected raw scores must have a positive sum")
    cutoff_indices = order[:, selected_experts : selected_experts + 1]
    return {
        "indices": selected,
        "weights": selected_raw / denominator,
        "cutoffs": torch.gather(biased, 1, cutoff_indices).squeeze(1),
        "cutoff_indices": cutoff_indices.squeeze(1),
    }


def qb_required_bias(scores, current_bias, selected_experts):
    """Return required biases r_ij = alpha_i - s_ij from one current step."""

    scores, current_bias = _routing_inputs(scores, current_bias, selected_experts)
    routes = quantile_routes(scores, current_bias, selected_experts)
    return routes["cutoffs"].unsqueeze(1) - scores


def exact_quantile_bias(scores, current_bias, selected_experts):
    """Evaluate K3 Eq. 14 with its exact (q+1)-th order statistic."""

    scores, current_bias = _routing_inputs(scores, current_bias, selected_experts)
    tokens, experts = scores.shape
    if (tokens * selected_experts) % experts:
        raise ValueError("exact QB requires integral target load m*k/n")
    target_load = tokens * selected_experts // experts
    if not 0 < target_load < tokens:
        raise ValueError("exact QB target load must be strictly between zero and m")
    required = qb_required_bias(scores, current_bias, selected_experts)
    uncentered = torch.sort(required, dim=0, stable=True).values[target_load]
    centered = uncentered - uncentered.mean()
    return {
        "bias": centered,
        "uncentered_bias": uncentered,
        "required_bias": required,
        "target_load": target_load,
    }


def histogram_quantile_counts(required_bias, current_bias, bins=1000):
    """Histogram one microbatch of required biases using K3 Appendix D."""

    required_bias = _cpu_tensor(
        required_bias, "required_bias", dimensions=2, floating=True
    )
    current_bias = _cpu_tensor(current_bias, "current_bias", dimensions=1, floating=True)
    _finite(required_bias, "required_bias")
    _finite(current_bias, "current_bias")
    if required_bias.shape[1] != current_bias.shape[0] or required_bias.shape[0] < 1:
        raise ValueError("required biases and current bias must name the same expert pool")
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer of at least two")
    lower = current_bias.min() - 1.0
    upper = current_bias.max() + 1.0
    width = (upper - lower) / bins
    if (required_bias < lower).any().item() or (required_bias > upper).any().item():
        raise ValueError("required bias escaped the source-derived adaptive range")
    indices = torch.floor((required_bias - lower) / width).to(torch.int64)
    indices.clamp_(0, bins - 1)
    counts = torch.zeros((current_bias.shape[0], bins), dtype=torch.int64)
    expert_indices = torch.arange(current_bias.shape[0]).expand(required_bias.shape[0], -1)
    ones = torch.ones(required_bias.numel(), dtype=torch.int64)
    counts.index_put_(
        (expert_indices.reshape(-1), indices.reshape(-1)), ones, accumulate=True
    )
    return {"counts": counts, "lower": lower, "upper": upper, "bin_width": width}


def histogram_quantile_bias(counts, current_bias, selected_experts):
    """Recover Appendix D's pooled histogram estimate and mean-center it."""

    counts = _cpu_tensor(counts, "counts", dimensions=2, floating=False)
    current_bias = _cpu_tensor(current_bias, "current_bias", dimensions=1, floating=True)
    _finite(current_bias, "current_bias")
    experts, bins = counts.shape
    if (
        counts.dtype != torch.int64
        or experts != current_bias.shape[0]
        or bins < 2
        or (counts < 0).any().item()
    ):
        raise ValueError("histogram counts must be non-negative and match current experts")
    totals = counts.sum(dim=1)
    if (totals < 1).any().item() or not torch.equal(totals, totals[0].expand_as(totals)):
        raise ValueError("every expert histogram must contain the same non-zero token count")
    if (
        isinstance(selected_experts, bool)
        or not isinstance(selected_experts, int)
        or not 0 < selected_experts < experts
    ):
        raise ValueError("selected_experts must be an integer between one and n-1")
    tokens = int(totals[0].item())
    target = tokens * selected_experts / experts
    target_ceiling = torch.ceil(torch.tensor(target)).to(torch.int64)
    cumulative = counts.cumsum(dim=1)
    selected_bins = (cumulative >= target_ceiling).to(torch.int64).argmax(dim=1)
    before_bins = (selected_bins - 1).clamp_min(0)
    before = torch.gather(cumulative, 1, before_bins.unsqueeze(1)).squeeze(1)
    before = torch.where(selected_bins == 0, torch.zeros_like(before), before)
    inside = torch.gather(counts, 1, selected_bins.unsqueeze(1)).squeeze(1)
    fraction = ((target - before.to(torch.float64)) / inside.to(torch.float64)).clamp(0, 1)
    lower = current_bias.min().to(torch.float64) - 1.0
    width = (current_bias.max().to(torch.float64) - current_bias.min().to(torch.float64) + 2.0) / bins
    uncentered = lower + (selected_bins.to(torch.float64) + fraction) * width
    centered = uncentered - uncentered.mean()
    return {
        "bias": centered,
        "uncentered_bias": uncentered,
        "target_load": target,
        "selected_bins": selected_bins,
        "bin_width": width,
    }
