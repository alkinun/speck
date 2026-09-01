"""Estimate context-dependent compute and resident model-state budgets."""

import math

import torch

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "int8": torch.int8,
}


def estimate_context_budget(
    model,
    lengths,
    *,
    effective_tflops,
    h100_hours,
    weight_bits=16,
    kv_cache_dtype="bfloat16",
):
    if not lengths or any(length < 1 for length in lengths):
        raise ValueError("budget lengths must be positive")
    if tuple(sorted(set(lengths))) != tuple(lengths):
        raise ValueError("budget lengths must be sorted and unique")
    numeric = (effective_tflops, h100_hours, weight_bits)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in numeric):
        raise ValueError("budget compute and precision values must be numeric")
    if not math.isfinite(effective_tflops) or effective_tflops <= 0:
        raise ValueError("effective TFLOP/s must be positive and finite")
    if not math.isfinite(h100_hours) or h100_hours <= 0:
        raise ValueError("H100-hours must be positive and finite")
    if not math.isfinite(weight_bits) or weight_bits <= 0:
        raise ValueError("weight bits must be positive and finite")
    if kv_cache_dtype not in DTYPES:
        raise ValueError("unsupported budget KV cache dtype")
    dtype = DTYPES[kv_cache_dtype]
    parameters = model.parameter_count()
    weight_bytes = math.ceil(parameters * weight_bits / 8)
    points = []
    for length in lengths:
        flops = model.flops_per_token(length)
        tokens_per_hour = effective_tflops * 1e12 * 3_600 / flops
        with torch.device("meta"):
            state = model.state(length=length, device="meta", kv_cache_dtype=dtype)
        memory = state.memory_report()
        points.append(
            {
                "length": length,
                "training_flops_per_token": flops,
                "tokens_per_h100_hour": tokens_per_hour,
                "tokens_in_budget": tokens_per_hour * h100_hours,
                "state_bytes": memory["total_bytes"],
                "state_by_kind": memory["by_kind"],
                "weights_plus_state_bytes": weight_bytes + memory["total_bytes"],
            }
        )
    base_flops = points[0]["training_flops_per_token"]
    for point in points:
        point["compute_multiple_vs_shortest"] = point["training_flops_per_token"] / base_flops
    return {
        "parameters": parameters,
        "weight_bits": weight_bits,
        "weight_bytes": weight_bytes,
        "kv_cache_dtype": kv_cache_dtype,
        "effective_tflops": effective_tflops,
        "h100_hours": h100_hours,
        "points": points,
    }
