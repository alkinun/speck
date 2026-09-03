"""Qualify FLA KDA outputs, gradients, cached recurrence, determinism, and throughput."""

import argparse
import importlib.metadata
import inspect
import json
import os
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from speck.long_context import parse_lengths
from speck.model import torch_kimi_delta_rule

PINNED_FLA_VERSION = "0.5.0"


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=parse_lengths, default=parse_lengths("64,512,4096"))
    parser.add_argument("--gradient-length", type=int, default=64)
    parser.add_argument("--decode-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--key-heads", type=int, default=3)
    parser.add_argument("--value-heads", type=int, default=6)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--forward-atol", type=float, default=0.02)
    parser.add_argument("--gradient-atol", type=float, default=0.05)
    parser.add_argument("--decode-atol", type=float, default=0.02)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def maximum_error(actual, expected):
    return (actual.float() - expected.float()).abs().max().item()


def command_output(command):
    result = subprocess.run(command, capture_output=True, check=False, text=True)
    return result.stdout.strip() or None


def inputs(args, length, requires_grad=False):
    key_shape = (args.batch_size, length, args.key_heads, args.head_dim)
    value_shape = (args.batch_size, length, args.value_heads, args.head_dim)
    query = torch.randn(*key_shape, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn(*value_shape, device="cuda", dtype=torch.bfloat16)
    log_decay = -torch.rand(*value_shape, device="cuda", dtype=torch.float32)
    beta = torch.rand(*value_shape[:-1], device="cuda", dtype=torch.bfloat16)
    values = (query, key, value, log_decay, beta)
    if requires_grad:
        values = tuple(value.detach().requires_grad_() for value in values)
    return values


def fla_operations():
    try:
        from fla.ops.kda import chunk_kda, fused_recurrent_kda
    except ImportError as error:
        raise RuntimeError(
            "flash-linear-attention is unavailable; install the gpu and linear extras"
        ) from error
    required = {
        "q",
        "k",
        "v",
        "g",
        "beta",
        "initial_state",
        "output_final_state",
        "use_qk_l2norm_in_kernel",
    }
    signatures = {
        name: str(inspect.signature(operation))
        for name, operation in {
            "chunk_kda": chunk_kda,
            "fused_recurrent_kda": fused_recurrent_kda,
        }.items()
    }
    for name, operation in {
        "chunk_kda": chunk_kda,
        "fused_recurrent_kda": fused_recurrent_kda,
    }.items():
        missing = required - set(inspect.signature(operation).parameters)
        if missing:
            raise RuntimeError(f"FLA {name} API is missing required parameters: {sorted(missing)}")
    return chunk_kda, fused_recurrent_kda, signatures


def fla_operation(operation, values, initial_state=None):
    query, key, value, log_decay, beta = values
    return operation(
        q=query,
        k=key,
        v=value,
        g=log_decay,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )


def reference_operation(values, initial_state=None):
    return torch_kimi_delta_rule(*values, initial_state=initial_state)


@torch.inference_mode()
def forward_case(args, operation, length):
    values = inputs(args, length)
    expected_output, expected_state = reference_operation(values)
    actual_output, actual_state = fla_operation(operation, values)
    repeated_output, repeated_state = fla_operation(operation, values)
    torch.cuda.synchronize()
    for _ in range(args.warmups):
        fla_operation(operation, values)
    torch.cuda.synchronize()
    durations = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        fla_operation(operation, values)
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


def gradient_case(args, operation):
    source = inputs(args, args.gradient_length)
    reference_values = tuple(value.detach().clone().requires_grad_() for value in source)
    actual_values = tuple(value.detach().clone().requires_grad_() for value in source)
    reference_output, reference_state = reference_operation(reference_values)
    reference_loss = reference_output.float().square().mean() + reference_state.square().mean()
    reference_loss.backward()
    actual_output, actual_state = fla_operation(operation, actual_values)
    actual_loss = actual_output.float().square().mean() + actual_state.float().square().mean()
    actual_loss.backward()
    names = ("query", "key", "value", "log_decay", "beta")
    return {
        name: maximum_error(actual.grad, expected.grad)
        for name, actual, expected in zip(names, actual_values, reference_values)
    }


@torch.inference_mode()
def decode_case(args, chunk_operation, recurrent_operation):
    values = inputs(args, args.decode_length)
    expected_output, expected_state = fla_operation(chunk_operation, values)
    outputs = []
    state = None
    for index in range(args.decode_length):
        token = tuple(value[:, index : index + 1] for value in values)
        output, state = fla_operation(recurrent_operation, token, initial_state=state)
        outputs.append(output)
    actual_output = torch.cat(outputs, dim=1)
    return {
        "length": args.decode_length,
        "output_max_abs_error": maximum_error(actual_output, expected_output),
        "state_max_abs_error": maximum_error(state, expected_state),
    }


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("KDA kernel qualification requires CUDA")
    dimensions = (
        args.gradient_length,
        args.decode_length,
        args.batch_size,
        args.key_heads,
        args.value_heads,
        args.head_dim,
        args.repeats,
    )
    if min(dimensions) < 1 or args.warmups < 0:
        raise ValueError("kernel dimensions and repetitions must be positive")
    if args.value_heads % args.key_heads:
        raise ValueError("value heads must be divisible by key heads")
    installed_version = importlib.metadata.version("flash-linear-attention")
    if installed_version != PINNED_FLA_VERSION:
        raise RuntimeError(
            f"KDA qualification requires flash-linear-attention=={PINNED_FLA_VERSION}, "
            f"found {installed_version}"
        )
    chunk_operation, recurrent_operation, signatures = fla_operations()
    torch.manual_seed(42)
    forward = [forward_case(args, chunk_operation, length) for length in args.lengths]
    gradients = gradient_case(args, chunk_operation)
    decode = decode_case(args, chunk_operation, recurrent_operation)
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
    for key in ("output_max_abs_error", "state_max_abs_error"):
        if decode[key] > args.decode_atol:
            failures.append(f"decode {key}={decode[key]:.6g}")
    report = {
        "format": "speck_kda_kernel_qualification",
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": not failures,
        "failures": failures,
        "forward_atol": args.forward_atol,
        "gradient_atol": args.gradient_atol,
        "decode_atol": args.decode_atol,
        "forward": forward,
        "gradient_max_abs_error": gradients,
        "decode": decode,
        "configuration": {
            "lengths": list(args.lengths),
            "gradient_length": args.gradient_length,
            "decode_length": args.decode_length,
            "batch_size": args.batch_size,
            "key_heads": args.key_heads,
            "value_heads": args.value_heads,
            "head_dim": args.head_dim,
            "warmups": args.warmups,
            "repeats": args.repeats,
        },
        "api": {
            "beta_activation": "external",
            "gate_parameterization": "external log decay",
            "signatures": signatures,
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
            "flash_linear_attention": installed_version,
        },
        "source": {
            "git_revision": command_output(["git", "rev-parse", "HEAD"]),
            "git_dirty": bool(command_output(["git", "status", "--porcelain"])),
        },
    }
    atomic_json(args.output, report)
    if failures:
        raise RuntimeError("KDA kernel qualification failed: " + "; ".join(failures))
    return report


def main(argv=None):
    report = run(arguments(argv))
    print(f"qualified {len(report['forward'])} lengths on {report['hardware']['device']}")


if __name__ == "__main__":
    main()
