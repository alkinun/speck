"""Qualify FLA Gated DeltaNet outputs, gradients, determinism, and throughput."""

import argparse
import importlib.metadata
import json
import os
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from speck.long_context import parse_lengths
from speck.model import torch_gated_delta_rule


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=parse_lengths, default=parse_lengths("64,512,4096"))
    parser.add_argument("--gradient-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--key-dim", type=int, default=64)
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--forward-atol", type=float, default=0.02)
    parser.add_argument("--gradient-atol", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def maximum_error(actual, expected):
    return (actual.float() - expected.float()).abs().max().item()


def command_output(command):
    result = subprocess.run(command, capture_output=True, check=False, text=True)
    return result.stdout.strip() or None


def inputs(args, length, requires_grad=False):
    shape = (args.batch_size, length, args.heads)
    query = torch.randn(*shape, args.key_dim, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn(*shape, args.value_dim, device="cuda", dtype=torch.bfloat16)
    log_decay = -torch.rand(*shape, device="cuda", dtype=torch.float32)
    beta = torch.rand(*shape, device="cuda", dtype=torch.bfloat16)
    values = (query, key, value, log_decay, beta)
    if requires_grad:
        values = tuple(value.detach().requires_grad_() for value in values)
    return values


def fla_operation(values):
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    except ImportError as error:
        raise RuntimeError(
            "flash-linear-attention is unavailable; install the gpu and linear extras"
        ) from error
    query, key, value, log_decay, beta = values
    return chunk_gated_delta_rule(
        query,
        key,
        value,
        g=log_decay,
        beta=beta,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )


def reference_operation(values):
    return torch_gated_delta_rule(*values)


@torch.inference_mode()
def forward_case(args, length):
    values = inputs(args, length)
    expected_output, expected_state = reference_operation(values)
    actual_output, actual_state = fla_operation(values)
    repeated_output, repeated_state = fla_operation(values)
    torch.cuda.synchronize()
    for _ in range(args.warmups):
        fla_operation(values)
    torch.cuda.synchronize()
    durations = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        fla_operation(values)
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - started)
    median = statistics.median(durations)
    return {
        "length": length,
        "output_max_abs_error": maximum_error(actual_output, expected_output),
        "state_max_abs_error": maximum_error(actual_state, expected_state),
        "repeat_output_max_abs_error": maximum_error(repeated_output, actual_output),
        "repeat_state_max_abs_error": maximum_error(repeated_state, actual_state),
        "median_seconds": median,
        "tokens_per_second": args.batch_size * length / median,
        "durations_seconds": durations,
    }


def gradient_case(args):
    source = inputs(args, args.gradient_length)
    reference_values = tuple(value.detach().clone().requires_grad_() for value in source)
    actual_values = tuple(value.detach().clone().requires_grad_() for value in source)
    reference_output, reference_state = reference_operation(reference_values)
    reference_loss = reference_output.float().square().mean() + reference_state.square().mean()
    reference_loss.backward()
    actual_output, actual_state = fla_operation(actual_values)
    actual_loss = actual_output.float().square().mean() + actual_state.float().square().mean()
    actual_loss.backward()
    names = ("query", "key", "value", "log_decay", "beta")
    return {
        name: maximum_error(actual.grad, expected.grad)
        for name, actual, expected in zip(names, actual_values, reference_values)
    }


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Gated DeltaNet kernel qualification requires CUDA")
    if (
        min(
            args.gradient_length,
            args.batch_size,
            args.heads,
            args.key_dim,
            args.value_dim,
            args.repeats,
        )
        < 1
        or args.warmups < 0
    ):
        raise ValueError("kernel dimensions and repetitions must be positive")
    torch.manual_seed(42)
    forward = [forward_case(args, length) for length in args.lengths]
    gradients = gradient_case(args)
    failures = []
    for case in forward:
        for key in ("output_max_abs_error", "state_max_abs_error"):
            if case[key] > args.forward_atol:
                failures.append(f"length {case['length']} {key}={case[key]:.6g}")
        for key in ("repeat_output_max_abs_error", "repeat_state_max_abs_error"):
            if case[key] != 0:
                failures.append(f"length {case['length']} nondeterministic {key}={case[key]:.6g}")
    failures.extend(
        f"gradient {name}={error:.6g}"
        for name, error in gradients.items()
        if error > args.gradient_atol
    )
    report = {
        "format": "speck_gdn_kernel_qualification",
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": not failures,
        "failures": failures,
        "forward_atol": args.forward_atol,
        "gradient_atol": args.gradient_atol,
        "forward": forward,
        "gradient_max_abs_error": gradients,
        "configuration": {
            "lengths": list(args.lengths),
            "gradient_length": args.gradient_length,
            "batch_size": args.batch_size,
            "heads": args.heads,
            "key_dim": args.key_dim,
            "value_dim": args.value_dim,
            "warmups": args.warmups,
            "repeats": args.repeats,
        },
        "hardware": {
            "device": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "driver": command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ]
            ),
        },
        "software": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "flash_linear_attention": importlib.metadata.version("flash-linear-attention"),
        },
        "source": {
            "git_revision": command_output(["git", "rev-parse", "HEAD"]),
            "git_dirty": bool(command_output(["git", "status", "--porcelain"])),
        },
    }
    atomic_json(args.output, report)
    if failures:
        raise RuntimeError("Gated DeltaNet kernel qualification failed: " + "; ".join(failures))
    return report


def main(argv=None):
    report = run(arguments(argv))
    print(f"qualified {len(report['forward'])} lengths on {report['hardware']['device']}")


if __name__ == "__main__":
    main()
