"""Build and score deterministic long-context diagnostics."""

import math
import random
import time
from collections import defaultdict

import torch


def parse_lengths(value):
    try:
        lengths = tuple(int(item) for item in value.split(","))
    except (AttributeError, ValueError) as error:
        raise ValueError("lengths must be comma-separated integers") from error
    if not lengths or any(length < 32 for length in lengths):
        raise ValueError("context lengths must be at least 32 tokens")
    if tuple(sorted(set(lengths))) != lengths:
        raise ValueError("context lengths must be sorted and unique")
    return lengths


def effective_length(curve, threshold=0.85, metric="exact_match"):
    """Return the longest length retaining the threshold fraction of short performance."""

    if not 0 < threshold <= 1:
        raise ValueError("effective-length threshold must be in (0, 1]")
    if not curve:
        return None
    baseline = curve[0][metric]
    if baseline <= 0:
        return None
    cutoff = threshold * baseline
    retained = [point["length"] for point in curve if point[metric] >= cutoff]
    return max(retained) if retained else None


def _repeat_to_length(pattern, length):
    if not pattern:
        raise ValueError("filler text must produce at least one token")
    repetitions = (length + len(pattern) - 1) // len(pattern)
    return (pattern * repetitions)[:length]


def build_passkey_case(tokenizer, length, seed, depth):
    """Build an exact-token-length literal retrieval diagnostic at a controlled depth."""

    if not 0 <= depth <= 1:
        raise ValueError("needle depth must be in [0, 1]")
    generator = random.Random(seed)
    label = f"archive-{generator.randrange(100_000, 1_000_000)}"
    answer = str(generator.randrange(100_000, 1_000_000))
    prefix = tokenizer.encode(
        "A long archive follows. Remember exact records and answer the final question.\n",
        bos=True,
    )
    needle = tokenizer.encode(f"\nThe access code for {label} is {answer}.\n")
    question = tokenizer.encode(f"\nQuestion: What is the access code for {label}?\nAnswer:")
    answer_tokens = tokenizer.encode(" " + answer)
    fixed = len(prefix) + len(needle) + len(question)
    if fixed > length:
        raise ValueError(f"context length {length} is too short for the diagnostic template")
    filler_length = length - fixed
    filler = _repeat_to_length(
        tokenizer.encode(
            "The archive contains ordinary reports, inventories, correspondence, and notes. "
        ),
        filler_length,
    )
    before = round(filler_length * depth)
    prompt = prefix + filler[:before] + needle + filler[before:] + question
    if len(prompt) != length:
        raise RuntimeError("long-context diagnostic did not reach its exact requested length")
    return {
        "task": "passkey",
        "length": length,
        "depth": depth,
        "seed": seed,
        "prompt_tokens": prompt,
        "answer_tokens": answer_tokens,
        "answer": answer,
        "label": label,
    }


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def evaluate_case(model, case, device=None, state_dtype=None, kv_cache_dtype=None):
    """Score an answer autoregressively without materializing sequence-wide vocabulary logits."""

    parameter = next(model.parameters())
    device = torch.device(device or parameter.device)
    prompt = torch.tensor([case["prompt_tokens"]], device=device)
    answers = case["answer_tokens"]
    state = model.state(
        length=prompt.size(1) + len(answers),
        device=device,
        dtype=state_dtype,
        kv_cache_dtype=kv_cache_dtype,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    started = time.perf_counter()
    logits = model(prompt, state=state, last_token_only=True)[:, -1]
    _synchronize(device)
    prefill_seconds = time.perf_counter() - started
    predictions = []
    log_probabilities = []
    decode_started = time.perf_counter()
    for token_id in answers:
        log_probabilities.append(torch.log_softmax(logits.float(), dim=-1)[0, token_id].item())
        predictions.append(logits.argmax(dim=-1).item())
        token = torch.tensor([[token_id]], device=device)
        logits = model(token, state=state, last_token_only=True)[:, -1]
    _synchronize(device)
    decode_seconds = time.perf_counter() - decode_started
    matched = sum(expected == actual for expected, actual in zip(answers, predictions))
    memory = state.memory_report()
    return {
        "task": case["task"],
        "length": case["length"],
        "depth": case["depth"],
        "seed": case["seed"],
        "answer_tokens": len(answers),
        "exact_match": float(matched == len(answers)),
        "token_accuracy": matched / len(answers),
        "mean_log_probability": sum(log_probabilities) / len(log_probabilities),
        "prefill_seconds": prefill_seconds,
        "prefill_tokens_per_second": case["length"] / prefill_seconds,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": len(answers) / decode_seconds,
        "state_memory": memory,
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }


def aggregate_results(results, threshold=0.85):
    grouped = defaultdict(list)
    for result in results:
        grouped[result["length"]].append(result)
    curve = []
    metrics = (
        "exact_match",
        "token_accuracy",
        "mean_log_probability",
        "prefill_seconds",
        "prefill_tokens_per_second",
        "decode_tokens_per_second",
    )
    for length, values in sorted(grouped.items()):
        point = {"length": length, "samples": len(values)}
        for metric in metrics:
            point[metric] = sum(value[metric] for value in values) / len(values)
        point["state_bytes"] = max(value["state_memory"]["total_bytes"] for value in values)
        point["state_by_kind"] = values[0]["state_memory"]["by_kind"]
        peak_values = [value["peak_allocated_bytes"] for value in values]
        point["peak_allocated_bytes"] = (
            max(value for value in peak_values if value is not None)
            if any(value is not None for value in peak_values)
            else None
        )
        curve.append(point)
    return {
        "curve": curve,
        "effective_length": effective_length(curve, threshold),
        "effective_length_threshold": threshold,
        "short_context_baseline": curve[0]["exact_match"] if curve else None,
    }


def validate_eval_settings(settings):
    required = {"lengths", "depths", "samples_per_depth", "effective_threshold"}
    allowed = required | {"kv_cache_dtype"}
    if not required <= set(settings) or set(settings) - allowed:
        raise ValueError("long-context settings have missing or unknown fields")
    lengths = settings["lengths"]
    if not isinstance(lengths, list):
        raise ValueError("long-context lengths must be a list")
    parse_lengths(",".join(map(str, lengths)))
    depths = settings["depths"]
    if (
        not isinstance(depths, list)
        or not depths
        or any(
            isinstance(depth, bool) or not isinstance(depth, (int, float)) or not 0 <= depth <= 1
            for depth in depths
        )
    ):
        raise ValueError("long-context depths must be numbers in [0, 1]")
    samples = settings["samples_per_depth"]
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("samples_per_depth must be a positive integer")
    threshold = settings["effective_threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("effective_threshold must be numeric")
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("effective_threshold must be in (0, 1]")
    kv_cache_dtype = settings.get("kv_cache_dtype", "bfloat16")
    if kv_cache_dtype not in {"bfloat16", "float16", "float32", "int8"}:
        raise ValueError("unsupported long-context KV cache dtype")
    return {
        "lengths": tuple(lengths),
        "depths": tuple(float(depth) for depth in depths),
        "samples_per_depth": samples,
        "effective_threshold": float(threshold),
        "kv_cache_dtype": kv_cache_dtype,
    }
