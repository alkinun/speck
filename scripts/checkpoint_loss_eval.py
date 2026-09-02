"""Evaluate a checkpoint's causal loss on an explicitly selected packed dataset."""

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from scripts.base_train import validate
from scripts.infer import load_checkpoint_model
from speck.checkpoint import checkpoint_identity, latest
from speck.common import base_dir
from speck.config import load_experiment
from speck.dataloader import manifest_fingerprint, packed_loader
from speck.dataset import load_manifest, resolve_data_dir
from speck.tokenizer import get_tokenizer


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_experiment", type=Path)
    parser.add_argument("--data-experiment", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--eval-tokens", type=int, default=20_000_000)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--loss-backend", choices=("torch", "liger"), default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def run(args):
    model_configs = load_experiment(args.model_experiment, "tokenizer", "train")
    data_experiment = args.data_experiment or args.model_experiment
    data_configs = load_experiment(data_experiment, "data", "tokenizer")
    if data_configs["tokenizer"] != model_configs["tokenizer"]:
        raise ValueError("evaluation data tokenizer does not match the model")
    train = model_configs["train"]
    checkpoint_dir = args.checkpoint_dir or Path(
        train.get("output_dir") or Path(base_dir()) / "checkpoints" / train["run"]
    )
    step = args.step if args.step is not None else latest(checkpoint_dir)
    if step is None:
        raise FileNotFoundError(f"no checkpoint found in {checkpoint_dir}")
    sequence_length = _positive_integer(
        args.sequence_length or train["sequence_length"], "sequence length"
    )
    eval_tokens = _positive_integer(args.eval_tokens, "evaluation tokens")
    batch_size = _positive_integer(
        args.batch_size or train["device_batch_size"], "batch size"
    )
    loss_backend = args.loss_backend or train.get("loss_backend", "torch")
    device = torch.device(args.device)
    model, metadata = load_checkpoint_model(
        checkpoint_dir, step, device, loss_backend=loss_backend
    )
    if sequence_length > model.config.max_position_embeddings:
        raise ValueError("evaluation sequence exceeds the model context")
    tokenizer = get_tokenizer(**data_configs["tokenizer"])
    data_dir = resolve_data_dir(
        data_configs["data"].get("output_dir"), data_configs["data"].get("output_name")
    )
    manifest = load_manifest(data_dir)
    tokens_per_step = batch_size * sequence_length
    steps = max(1, min(eval_tokens, manifest["splits"]["val"]["tokens"]) // tokens_per_step)
    loader = packed_loader(
        tokenizer,
        batch_size,
        sequence_length,
        "val",
        device=device,
        data_dir=data_dir,
    )
    evaluated_tokens = steps * tokens_per_step
    eval_model = model if args.no_compile else torch.compile(model, dynamic=False)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    loss, source_losses = validate(
        eval_model,
        loader,
        steps,
        world_size=1,
        source_ids=tuple(source["id"] for source in manifest["sources"]),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    duration = time.perf_counter() - started
    report = {
        "format": "speck_checkpoint_loss_evaluation",
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_experiment": str(args.model_experiment.expanduser().resolve()),
        "checkpoint": checkpoint_identity(checkpoint_dir, step),
        "checkpoint_global_tokens": metadata.get("global_tokens"),
        "data_experiment": str(data_experiment.expanduser().resolve()),
        "data_directory": str(data_dir.expanduser().resolve()),
        "data_manifest": manifest_fingerprint(manifest),
        "sequence_length": sequence_length,
        "requested_eval_tokens": eval_tokens,
        "evaluated_tokens": evaluated_tokens,
        "batch_size": batch_size,
        "steps": steps,
        "loss_backend": loss_backend,
        "compiled": not args.no_compile,
        "loss": loss,
        "perplexity": math.exp(loss),
        "source_losses": source_losses,
        "seconds": duration,
        "tokens_per_second": evaluated_tokens / duration,
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "device": str(device),
        "torch_version": torch.__version__,
        "rope_scaling_factor": model.config.rope_scaling_factor,
    }
    output = (
        args.output
        or Path(base_dir())
        / "evaluations"
        / "checkpoint-loss"
        / train["run"]
        / f"{Path(data_experiment).name}-{sequence_length}-{step}.json"
    )
    atomic_json(output, report)
    return report


def main(argv=None):
    report = run(arguments(argv))
    print(
        f"loss {report['loss']:.5f} | perplexity {report['perplexity']:.2f} | "
        f"{report['evaluated_tokens']:,} tokens"
    )


if __name__ == "__main__":
    main()
