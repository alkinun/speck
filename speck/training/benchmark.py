"""Benchmark a Speck optimization step."""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.autograd import DeviceType

from speck.config import load_experiment
from speck.data.dataset import load_manifest
from speck.data.loader import manifest_fingerprint, packed_loader
from speck.evaluation.diagnostics import nearest_percentile as percentile
from speck.evaluation.diagnostics import synchronize
from speck.model import build_model
from speck.operations.runtime import COMPILE_OPTIONS, configure_determinism
from speck.tokenization.tokenizer import get_tokenizer
from speck.training.step import optimization_step

_COMPILE_MODE_OPTIONS = {
    "default": {},
    "reduce-overhead": {"triton.cudagraphs": True},
    # Unlike Torch's named modes, these omit coordinate-descent tuning; see COMPILE_OPTIONS.
    "max-autotune": {"max_autotune": True, "triton.cudagraphs": True},
    "max-autotune-no-cudagraphs": {"max_autotune": True},
}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment",
        help="experiment directory",
    )
    parser.add_argument(
        "--mode",
        choices=("compute", "end-to-end"),
        default="compute",
        help="benchmark synthetic compute or include packed-data loading (default: %(default)s)",
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.expanduser("~/.cache/speck/benchmark-200m"),
        help="packed dataset directory for end-to-end mode (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="benchmark device (default: CUDA when available, otherwise CPU)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of measured optimization steps (default: %(default)s)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
        help="number of warmup optimization steps (default: %(default)s)",
    )
    parser.add_argument(
        "--accumulation",
        type=int,
        default=None,
        help="gradient accumulation steps; defaults to the experiment configuration",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="device batch size; defaults to the experiment configuration",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="sequence length; defaults to the experiment configuration",
    )
    parser.add_argument(
        "--peak-tflops",
        type=float,
        default=None,
        help="theoretical peak TFLOPS used to calculate model FLOPs utilization",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed (default: %(default)s)",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="disable torch.compile",
    )
    parser.add_argument(
        "--compile-mode",
        default="max-autotune-no-cudagraphs",
        choices=tuple(_COMPILE_MODE_OPTIONS),
        help="torch.compile mode (default: %(default)s)",
    )
    parser.add_argument(
        "--loss-backend",
        choices=("torch", "liger"),
        default="torch",
        help="linear cross-entropy implementation (default: %(default)s)",
    )
    parser.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override train.json activation checkpointing for this benchmark",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="select reproducible kernels, matching production recipes (default: %(default)s)",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="record torch._dynamo.explain graph and graph-break counts",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="optional path for a chrome trace; profiled steps run after the timed steps",
    )
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=3,
        help="number of profiled steps taken after timing (default: %(default)s)",
    )
    parser.add_argument(
        "--memory-snapshot",
        default=None,
        help="optional path for a CUDA memory snapshot recorded after the timed steps",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="short label identifying this configuration in a sweep",
    )
    parser.add_argument(
        "--boundary",
        default=None,
        help="prose statement constraining what this measurement may be used for",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="optional path for the JSON benchmark report",
    )
    return parser.parse_args(argv)


def package_versions():
    """Return installed versions of the kernel packages that affect throughput."""

    versions = {}
    for name in ("triton", "fla", "liger_kernel"):
        try:
            module = __import__(name)
        except Exception:
            versions[name] = None
        else:
            versions[name] = getattr(module, "__version__", "unknown")
    return versions


def device_telemetry(device):
    """Sample SM clock, temperature and power so throttled results can be rejected."""

    if device.type != "cuda":
        return None
    query = "clocks.sm,temperature.gpu,power.draw"
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        capture_output=True,
        check=False,
        text=True,
    )
    line = result.stdout.strip().splitlines()
    if not line:
        return None
    fields = [field.strip() for field in line[0].split(",")]
    if len(fields) != 3:
        return None
    return {
        "sm_clock_mhz": _optional_float(fields[0]),
        "temperature_c": _optional_float(fields[1]),
        "power_w": _optional_float(fields[2]),
    }


def _optional_float(text):
    try:
        return float(text)
    except ValueError:
        return None


_MIXER_MARKERS = ("chunkkda", "kda", "flash", "attention", "fused_recurrent")
_GEMM_MARKERS = ("gemm", "cutlass", "s16816", "s1688", "wgmma", "_mm", "mm_", "bmm", "dot")
_LAUNCH_MARKERS = ("command buffer", "cudalaunch", "cudastream", "cudaevent")


def classify_kernel(name):
    """Group a device kernel so the stopping rule can be evaluated mechanically.

    Mixer kernels are matched before GEMM because fused attention and KDA kernels
    carry matmul markers in their template arguments.
    """

    lowered = name.lower()
    if lowered.startswith("aten::") or lowered.startswith("autograd::"):
        return "operator"
    if lowered.startswith("optimizer.step") or "#" in name:
        return "annotation"
    if any(marker in lowered for marker in _LAUNCH_MARKERS):
        return "launch"
    if any(marker in lowered for marker in _MIXER_MARKERS):
        return "mixer"
    if any(marker in lowered for marker in _GEMM_MARKERS):
        return "gemm"
    if "memcpy" in lowered or "copy" in lowered:
        return "copy"
    if "elementwise" in lowered or "fill" in lowered or "vectorized" in lowered:
        return "pointwise"
    return "other"


def kernel_summary(profiler, limit=25):
    """Fold the profiler's hottest device kernels into the receipt.

    Only CUDA events contribute: CPU custom-autograd rows can carry device time
    for the very same kernels. Rows sharing a name are summed, since
    recording shapes splits one kernel across several entries. Launch-stall
    markers are reported separately rather than counted as kernel work.
    """

    merged = {}
    for event in profiler.key_averages():
        if event.device_type != DeviceType.CUDA:
            continue
        micros = getattr(event, "self_device_time_total", None)
        if micros is None:
            micros = getattr(event, "self_cuda_time_total", 0.0)
        if not micros:
            continue
        kind = classify_kernel(event.key)
        if kind in {"operator", "annotation"}:
            continue
        recorded = merged.setdefault(event.key, {"micros": 0.0, "count": 0, "kind": kind})
        recorded["micros"] += float(micros)
        recorded["count"] += int(event.count)
    kernels = [
        (value["micros"], key, value["count"], value["kind"])
        for key, value in merged.items()
        if value["kind"] != "launch"
    ]
    stalls = sum(value["micros"] for value in merged.values() if value["kind"] == "launch")
    kernels.sort(reverse=True)
    total = sum(micros for micros, _, _, _ in kernels)
    shares = {}
    for micros, _, _, kind in kernels:
        shares[kind] = shares.get(kind, 0.0) + micros
    return {
        "accounting": "CUDA events only; summed kernel time, not elapsed wall time or GPU utilization",
        "self_device_time_total_us": total,
        "launch_stall_us": stalls,
        "useful_percent": 100.0 * (shares.get("gemm", 0.0) + shares.get("mixer", 0.0)) / total
        if total
        else None,
        "category_percent": {
            kind: 100.0 * value / total if total else None for kind, value in sorted(shares.items())
        },
        "top_kernels": [
            {
                "name": key,
                "kind": kind,
                "self_device_time_us": micros,
                "count": count,
                "percent": 100.0 * micros / total if total else None,
            }
            for micros, key, count, kind in kernels[:limit]
        ],
    }


def resolve_activation_checkpointing(train, override):
    configured = train.get("activation_checkpointing", False)
    if not isinstance(configured, bool):
        raise ValueError("train activation_checkpointing must be boolean")
    if override is not None and not isinstance(override, bool):
        raise ValueError("activation checkpointing override must be boolean or null")
    return configured if override is None else override


def config_fingerprint(configs):
    payload = json.dumps(configs, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def git_revision():
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() or None


def git_dirty():
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        check=False,
        text=True,
    )
    return bool(result.stdout.strip())


def synthetic_loader(batch_size, sequence_length, vocab_size, device):
    tokens = torch.randint(vocab_size, (batch_size, sequence_length + 1), device=device)
    batch = (tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous(), None)
    while True:
        yield batch


def run(args):
    if args.steps < 1 or args.warmup_steps < 0:
        raise ValueError("steps must be positive and warmup steps cannot be negative")
    if args.profile_steps < 1:
        raise ValueError("profile steps must be positive")
    configure_determinism(args.deterministic)
    configs = load_experiment(args.experiment, "tokenizer", "model", "train")
    tokenizer = get_tokenizer(**configs["tokenizer"])
    train = configs["train"]
    activation_checkpointing = resolve_activation_checkpointing(
        train, args.activation_checkpointing
    )
    batch_size = args.batch_size or train["device_batch_size"]
    sequence_length = args.sequence_length or train["sequence_length"]
    micro_tokens = batch_size * sequence_length
    if args.accumulation is None:
        if train["batch_tokens"] % micro_tokens:
            raise ValueError("batch tokens must be divisible by benchmark microbatch tokens")
        accumulation = train["batch_tokens"] // micro_tokens
    else:
        accumulation = args.accumulation
    if accumulation < 1:
        raise ValueError("accumulation must be positive")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    model = build_model(
        configs["model"],
        tokenizer.vocab_size,
        tokenizer.bos_id,
        tokenizer.eos_id,
        loss_backend=args.loss_backend,
    ).to(device)
    model.init_weights()
    model.set_gradient_checkpointing(activation_checkpointing)
    parameters = tuple(model.parameters())
    optimizer = model.optimizer(train["lr"], train["weight_decay"], train["optimizer"])
    train_model = (
        model
        if args.no_compile
        else torch.compile(
            model,
            dynamic=False,
            options={
                **_COMPILE_MODE_OPTIONS[args.compile_mode],
                "aggressive_fusion": True,
            },
        )
    )
    optimizer_step_compiled = not args.no_compile and hasattr(optimizer, "compile_step")
    if optimizer_step_compiled:
        optimizer.compile_step(COMPILE_OPTIONS)
    cudagraphs = not args.no_compile and args.compile_mode in {"reduce-overhead", "max-autotune"}

    manifest_hash = None
    if args.mode == "compute":
        loader = synthetic_loader(batch_size, sequence_length, tokenizer.vocab_size, device)
    else:
        manifest = load_manifest(args.data_dir)
        manifest_hash = manifest_fingerprint(manifest)
        loader = packed_loader(
            tokenizer,
            batch_size,
            sequence_length,
            "train",
            device=device,
            data_dir=args.data_dir,
        )
    batch = next(loader)
    telemetry_started = device_telemetry(device)

    explain = None
    if args.explain and not args.no_compile:
        explanation = torch._dynamo.explain(train_model)(
            batch[0],
            batch[1],
        )
        explain = {
            "graph_count": explanation.graph_count,
            "graph_break_count": explanation.graph_break_count,
            "op_count": explanation.op_count,
            "break_reasons": [
                str(getattr(reason, "reason", reason)) for reason in explanation.break_reasons
            ][:25],
        }
        torch._dynamo.reset()

    def step(probe=None):
        return optimization_step(
            train_model,
            parameters,
            optimizer,
            loader,
            batch,
            accumulation,
            train["grad_clip"],
            train["lr"],
            cudagraphs=cudagraphs,
            step_probe=probe,
        )

    started = time.perf_counter()
    for _ in range(args.warmup_steps):
        _, _, batch = step()
    synchronize(device)
    warmup_seconds = time.perf_counter() - started

    # Peak memory is split so the optimizer transient cannot mask the activation
    # footprint. They occupy the same allocator blocks at different times, so the
    # peak counter is reset every step and each phase keeps its own running maximum.
    cuda = device.type == "cuda"
    peak_after_backward = 0
    peak_after_step = 0
    peak_reserved = 0

    def record_backward_peak():
        nonlocal peak_after_backward
        if cuda:
            peak_after_backward = max(peak_after_backward, torch.cuda.max_memory_allocated(device))

    durations = []
    losses = []
    for _ in range(args.steps):
        synchronize(device)
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        loss, _, batch = step(probe=record_backward_peak)
        synchronize(device)
        durations.append(time.perf_counter() - started)
        losses.append(float(loss))
        if cuda:
            peak_after_step = max(peak_after_step, torch.cuda.max_memory_allocated(device))
            peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(device))

    telemetry_finished = device_telemetry(device)

    profile_summary = None
    if args.profile:
        # Profiling perturbs timing, so it runs after the measured steps.
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        ) as profiler:
            for _ in range(args.profile_steps):
                _, _, batch = step()
            synchronize(device)
        path = Path(args.profile)
        path.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(path))
        profile_summary = {"trace": str(path), **kernel_summary(profiler)}

    if args.memory_snapshot and device.type == "cuda":
        torch.cuda.memory._record_memory_history()
        for _ in range(2):
            _, _, batch = step()
        synchronize(device)
        snapshot = Path(args.memory_snapshot)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._dump_snapshot(str(snapshot))
        torch.cuda.memory._record_memory_history(enabled=None)

    tokens_per_step = batch_size * sequence_length * accumulation
    total_seconds = sum(durations)
    tokens_per_second = tokens_per_step * args.steps / total_seconds
    tflops = model.flops_per_token(sequence_length) * tokens_per_second / 1e12
    median = statistics.median(durations)
    p90 = percentile(durations, 0.9)
    duration_stddev = statistics.stdev(durations) if len(durations) > 1 else 0.0
    step_rates = [tokens_per_step / duration for duration in durations]
    rate_median = statistics.median(step_rates)
    rate_p10 = percentile(step_rates, 0.1)
    rate_p90 = percentile(step_rates, 0.9)
    result = {
        "format": "speck_throughput_probe",
        "format_version": 2,
        "label": args.label,
        "boundary": args.boundary,
        "benchmark": {
            "mode": args.mode,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "warmup_seconds": warmup_seconds,
            "peak_tflops": args.peak_tflops,
            "compiled": not args.no_compile,
            "compile_mode": None if args.no_compile else args.compile_mode,
            "aggressive_fusion": not args.no_compile,
            "loss_backend": args.loss_backend,
            "activation_checkpointing": activation_checkpointing,
            "deterministic": args.deterministic,
            "optimizer": train["optimizer"],
            "optimizer_step_compiled": optimizer_step_compiled,
            "seed": args.seed,
        },
        "compile": explain,
        "profile": profile_summary,
        "quality": {
            "stable": bool(p90 / median < 1.05) if median else None,
            "p90_over_median": p90 / median if median else None,
            "step_seconds_stddev": duration_stddev,
            "step_seconds_cv": duration_stddev / statistics.mean(durations)
            if durations and statistics.mean(durations)
            else None,
        },
        "geometry": {
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "accumulation": accumulation,
            "tokens_per_step": tokens_per_step,
        },
        "performance": {
            "tokens_per_second": tokens_per_second,
            "tflops": tflops,
            "model_flops_utilization": tflops / args.peak_tflops if args.peak_tflops else None,
            "step_seconds_mean": statistics.mean(durations),
            "step_seconds_median": median,
            "step_seconds_p90": p90,
            "step_seconds_min": min(durations),
            "step_seconds_max": max(durations),
            "tokens_per_second_median": rate_median,
            "tokens_per_second_p10": rate_p10,
            "tokens_per_second_p90": rate_p90,
            "loss_first": losses[0],
            "loss_last": losses[-1],
        },
        "memory": {
            "peak_allocated_bytes": peak_after_step if cuda else None,
            "peak_reserved_bytes": peak_reserved if cuda else None,
            "peak_after_backward_bytes": peak_after_backward if cuda else None,
            "peak_after_step_bytes": peak_after_step if cuda else None,
        },
        "environment": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "device_capability": list(torch.cuda.get_device_capability(device))
            if device.type == "cuda"
            else None,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "packages": package_versions(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "kernel_caches": {
                "inductor": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
                "triton": os.environ.get("TRITON_CACHE_DIR"),
            },
            "telemetry_started": telemetry_started,
            "telemetry_finished": telemetry_finished,
            "git_revision": git_revision(),
            "git_dirty": git_dirty(),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        },
        "model": {
            "parameters": model.parameter_count(),
            "flops_per_token": model.flops_per_token(sequence_length),
            "flops_convention": (
                "6 * linear operation estimate plus attention/recurrent estimate; includes "
                "the vocabulary projection, excludes optimizer work and activation-recompute "
                "overhead"
            ),
        },
        "experiment": {
            "path": str(Path(args.experiment).resolve()),
            "fingerprint": config_fingerprint(configs),
            "manifest": manifest_hash,
        },
    }
    return result


def main():
    args = arguments()
    result = run(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
