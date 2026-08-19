"""measure the real speck optimization step."""

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

from speck.config import load_experiment
from speck.dataloader import manifest_fingerprint, packed_loader
from speck.dataset import load_manifest
from speck.model import build_model
from speck.tokenizer import get_tokenizer
from speck.train import optimization_step


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", nargs="?", default="experiments/speck-50m")
    parser.add_argument("--mode", choices=("compute", "end-to-end"), default="compute")
    parser.add_argument("--data-dir", default=os.path.expanduser("~/.cache/speck/benchmark-50m"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--accumulation", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--peak-tflops", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


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


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run(args):
    if args.steps < 1 or args.warmup_steps < 0:
        raise ValueError("steps must be positive and warmup steps cannot be negative")
    configs = load_experiment(args.experiment, "tokenizer", "model", "train")
    tokenizer = get_tokenizer(**configs["tokenizer"])
    train = configs["train"]
    batch_size = args.batch_size or train["device_batch_size"]
    sequence_length = args.sequence_length or train["sequence_length"]
    micro_tokens = batch_size * sequence_length
    if args.accumulation is None:
        if train["batch_tokens"] % micro_tokens:
            raise ValueError("batch tokens must be divisible by benchmark micro batch tokens")
        accumulation = train["batch_tokens"] // micro_tokens
    else:
        accumulation = args.accumulation
    if accumulation < 1:
        raise ValueError("accumulation must be positive")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    model = build_model(
        configs["model"], tokenizer.vocab_size, tokenizer.bos_id, tokenizer.eos_id
    ).to(device)
    model.init_weights()
    parameters = tuple(model.parameters())
    optimizer = model.optimizer(train["lr"], train["weight_decay"])
    train_model = model if args.no_compile else torch.compile(
        model, dynamic=False, mode="max-autotune-no-cudagraphs"
    )

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

    started = time.perf_counter()
    for _ in range(args.warmup_steps):
        _, _, batch = optimization_step(
            train_model,
            parameters,
            optimizer,
            loader,
            batch,
            accumulation,
            train["grad_clip"],
            train["lr"],
            weight_context=model.cached_weights,
        )
    synchronize(device)
    warmup_seconds = time.perf_counter() - started

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    durations = []
    losses = []
    for _ in range(args.steps):
        synchronize(device)
        started = time.perf_counter()
        loss, _, batch = optimization_step(
            train_model,
            parameters,
            optimizer,
            loader,
            batch,
            accumulation,
            train["grad_clip"],
            train["lr"],
            weight_context=model.cached_weights,
        )
        synchronize(device)
        durations.append(time.perf_counter() - started)
        losses.append(loss.item())

    tokens_per_step = batch_size * sequence_length * accumulation
    total_seconds = sum(durations)
    tokens_per_second = tokens_per_step * args.steps / total_seconds
    tflops = model.flops_per_token(sequence_length) * tokens_per_second / 1e12
    result = {
        "benchmark": {
            "mode": args.mode,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "warmup_seconds": warmup_seconds,
            "compiled": not args.no_compile,
            "seed": args.seed,
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
            "step_seconds_median": statistics.median(durations),
            "step_seconds_p90": percentile(durations, 0.9),
            "loss_first": losses[0],
            "loss_last": losses[-1],
        },
        "memory": {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
        },
        "environment": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "git_revision": git_revision(),
            "git_dirty": git_dirty(),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        },
        "model": {
            "parameters": model.parameter_count(),
            "flops_per_token": model.flops_per_token(sequence_length),
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
