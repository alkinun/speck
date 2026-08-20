"""compare training quality on a fixed packed token sample."""

import argparse
import json
import math
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.benchmark import config_fingerprint, git_dirty, git_revision
from speck.config import load_experiment
from speck.dataloader import manifest_fingerprint, packed_loader
from speck.dataset import load_manifest, verify_shards
from speck.model import build_model
from speck.tokenizer import get_tokenizer
from speck.train import lr_scale, optimization_step


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", nargs="?", default="experiments/speck-50m")
    parser.add_argument("--label", required=True)
    parser.add_argument("--data-dir", default=os.path.expanduser("~/.cache/speck/benchmark-50m"))
    parser.add_argument("--steps", type=int, default=95)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batch-tokens", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=24)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-tokens", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--optimizer", choices=("adamw", "muon"), default=None)
    parser.add_argument("--batch-curriculum", action="store_true")
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(model, tokenizer, data_dir, manifest, batch_size, sequence_length, device, token_limit=None):
    loader = packed_loader(
        tokenizer, batch_size, sequence_length, "val", device=device, data_dir=data_dir
    )
    tokens = manifest["splits"]["val"]["tokens"]
    if token_limit is not None:
        tokens = min(tokens, token_limit)
    steps = max(1, tokens // (batch_size * sequence_length))
    loss = torch.zeros((), device=device)
    model.eval()
    for _ in range(steps):
        inputs, targets, _ = next(loader)
        logits = model(inputs)
        loss += F.cross_entropy(logits.flatten(0, 1), targets.flatten())
    model.train()
    return (loss / steps).item(), steps * batch_size * sequence_length


def run(args):
    if args.steps < 1 or args.batch_size < 1 or args.eval_batch_size < 1:
        raise ValueError("steps and batch sizes must be positive")
    if args.eval_tokens is not None and args.eval_tokens < 1:
        raise ValueError("eval tokens must be positive")
    configs = load_experiment(args.experiment, "tokenizer", "model", "train")
    if args.model_config:
        configs["model"] = json.loads(Path(args.model_config).read_text(encoding="utf-8"))
    train = configs["train"]
    tokenizer = get_tokenizer(**configs["tokenizer"])
    manifest = load_manifest(args.data_dir)
    verify_shards(args.data_dir, manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    model = build_model(
        configs["model"], tokenizer.vocab_size, tokenizer.bos_id, tokenizer.eos_id
    ).to(device)
    model.init_weights()
    parameters = tuple(model.parameters())
    optimizer_name = args.optimizer or train["optimizer"]
    optimizer = model.optimizer(train["lr"], train["weight_decay"], optimizer_name)
    compiled_model = model if args.no_compile else torch.compile(
        model, dynamic=False, mode="max-autotune-no-cudagraphs"
    )

    sequence_length = train["sequence_length"]
    micro_tokens = args.batch_size * sequence_length
    fixed_batch_tokens = args.batch_tokens or train["batch_tokens"]
    if fixed_batch_tokens % micro_tokens:
        raise ValueError("batch tokens must be divisible by micro batch tokens")
    if args.batch_curriculum and args.batch_tokens:
        raise ValueError("batch tokens cannot override the batch curriculum")
    accumulation = fixed_batch_tokens // micro_tokens
    trained_tokens = args.steps * train["batch_tokens"]
    if manifest["splits"]["train"]["tokens"] <= trained_tokens + micro_tokens:
        raise ValueError("packed dataset is too small for this benchmark")

    loader = packed_loader(
        tokenizer,
        args.batch_size,
        sequence_length,
        "train",
        device=device,
        data_dir=args.data_dir,
    )
    batch = next(loader)
    validation = []

    def record_validation(step):
        loss, evaluated_tokens = evaluate(
            compiled_model,
            tokenizer,
            args.data_dir,
            manifest,
            args.eval_batch_size,
            sequence_length,
            device,
            args.eval_tokens,
        )
        validation.append({"step": step, "tokens": trained_tokens_done, "loss": loss})
        print(f"step {step:,} | validation loss {loss:.5f}")
        return evaluated_tokens

    trained_tokens_done = 0
    eval_tokens = record_validation(0)
    durations = []
    duration_tokens = []
    train_losses = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    step = 0
    next_eval_tokens = args.eval_every * train["batch_tokens"]
    quarter_tokens = math.ceil(args.steps / 4) * train["batch_tokens"]
    while trained_tokens_done < trained_tokens:
        if args.batch_curriculum and trained_tokens_done < quarter_tokens:
            batch_tokens = train["batch_tokens"] // 4
        elif args.batch_curriculum and trained_tokens_done < 2 * quarter_tokens:
            batch_tokens = train["batch_tokens"] // 2
        else:
            batch_tokens = fixed_batch_tokens
        accumulation = batch_tokens // micro_tokens
        schedule_step = trained_tokens_done / train["batch_tokens"]
        scale = lr_scale(schedule_step, args.steps, args.warmup_steps, train["min_lr"])
        synchronize(device)
        started = time.perf_counter()
        loss, _, batch = optimization_step(
            compiled_model,
            parameters,
            optimizer,
            loader,
            batch,
            accumulation,
            train["grad_clip"],
            train["lr"] * scale,
        )
        synchronize(device)
        durations.append(time.perf_counter() - started)
        duration_tokens.append(batch_tokens)
        train_losses.append(loss.item())
        step += 1
        trained_tokens_done += batch_tokens
        if trained_tokens_done >= next_eval_tokens or trained_tokens_done == trained_tokens:
            record_validation(step)
            next_eval_tokens += args.eval_every * train["batch_tokens"]

    measured_seconds = sum(durations[1:])
    measured_tokens = sum(duration_tokens[1:])
    result = {
        "benchmark": {
            "kind": "training_quality",
            "label": args.label,
            "steps": step,
            "seed": args.seed,
            "compiled": not args.no_compile,
            "loss": "torch",
            "optimizer": optimizer_name,
            "batch_curriculum": args.batch_curriculum,
        },
        "geometry": {
            "batch_size": args.batch_size,
            "sequence_length": sequence_length,
            "accumulation": accumulation,
            "tokens_per_step": fixed_batch_tokens,
            "trained_tokens": trained_tokens,
            "eval_batch_size": args.eval_batch_size,
            "eval_tokens": eval_tokens,
        },
        "quality": {
            "validation": validation,
            "train_loss_first": train_losses[0],
            "train_loss_last": train_losses[-1],
            "best_validation_loss": min(item["loss"] for item in validation),
            "final_perplexity": math.exp(min(validation[-1]["loss"], 20)),
        },
        "performance": {
            "tokens_per_second": measured_tokens / measured_seconds if measured_seconds else None,
            "step_seconds_median": statistics.median(durations[1:]) if len(durations) > 1 else durations[0],
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        },
        "dataset": {
            "path": str(Path(args.data_dir).resolve()),
            "manifest": manifest_fingerprint(manifest),
        },
        "model": {
            "parameters": model.parameter_count(),
            "config": configs["model"],
        },
        "experiment": {
            "path": str(Path(args.experiment).resolve()),
            "fingerprint": config_fingerprint(configs),
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
