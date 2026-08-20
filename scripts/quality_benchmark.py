"""compare training quality on a fixed packed token sample."""

import argparse
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

import torch

from scripts.benchmark import config_fingerprint, git_dirty, git_revision
from speck.config import load_experiment
from speck.dataloader import manifest_fingerprint
from speck.dataset import load_manifest, verify_shards
from speck.model import build_model
from speck.search.evaluate import QualitySettings, evaluate_quality
from speck.tokenizer import get_tokenizer


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", nargs="?", default="experiments/speck00-200m")
    parser.add_argument("--label", required=True)
    parser.add_argument("--data-dir", default=os.path.expanduser("~/.cache/speck/benchmark-200m"))
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
    with torch.device("meta"):
        model = build_model(
            configs["model"], tokenizer.vocab_size, tokenizer.bos_id, tokenizer.eos_id
        )
    config = model.config
    optimizer_name = args.optimizer or train["optimizer"]
    sequence_length = train["sequence_length"]
    fixed_batch_tokens = args.batch_tokens or train["batch_tokens"]
    if args.batch_curriculum and args.batch_tokens:
        raise ValueError("batch tokens cannot override the batch curriculum")
    trained_tokens = args.steps * train["batch_tokens"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    quality = evaluate_quality(
        config,
        tokenizer,
        QualitySettings(
            data_dir=args.data_dir,
            train_tokens=trained_tokens,
            batch_tokens=fixed_batch_tokens,
            device_batch_size=args.batch_size,
            sequence_length=sequence_length,
            eval_every_tokens=args.eval_every * train["batch_tokens"],
            eval_batch_size=args.eval_batch_size,
            eval_tokens=args.eval_tokens or manifest["splits"]["val"]["tokens"],
            lr=train["lr"],
            min_lr=train["min_lr"],
            warmup_steps=args.warmup_steps,
            weight_decay=train["weight_decay"],
            grad_clip=train["grad_clip"],
            optimizer=optimizer_name,
            compile=not args.no_compile,
            batch_curriculum=args.batch_curriculum,
        ),
        device,
        args.seed,
    )
    validation = quality["validation_curve"]
    train_curve = quality["train_curve"]
    durations = quality["performance"]["step_seconds"]
    result = {
        "benchmark": {
            "kind": "training_quality",
            "label": args.label,
            "steps": len(train_curve),
            "seed": args.seed,
            "compiled": not args.no_compile,
            "loss": "torch",
            "optimizer": optimizer_name,
            "batch_curriculum": args.batch_curriculum,
        },
        "geometry": {
            **quality["geometry"],
            "eval_batch_size": args.eval_batch_size,
        },
        "quality": {
            "validation": validation,
            "train": train_curve,
            "train_loss_first": train_curve[0]["loss"],
            "train_loss_last": train_curve[-1]["loss"],
            "best_validation_loss": min(item["loss"] for item in validation),
            "final_validation_loss": quality["validation_nll"],
            "final_perplexity": math.exp(min(validation[-1]["loss"], 20)),
        },
        "performance": {
            "tokens_per_second": quality["performance"]["tokens_per_second"],
            "step_seconds_median": statistics.median(durations),
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
