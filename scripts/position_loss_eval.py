"""Evaluate checkpoint loss by absolute sequence position and trailing region."""

import argparse
import json
import math
import os
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import torch

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
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--trailing-tokens", type=int, default=2_048)
    parser.add_argument("--loss-backend", choices=("torch", "liger"), default=None)
    parser.add_argument("--rope-scaling-factor", type=float, default=None)
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


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def position_ranges(sequence_length, bins):
    """Return contiguous near-equal position ranges covering the sequence exactly."""

    sequence_length = positive_integer(sequence_length, "sequence length")
    bins = positive_integer(bins, "bins")
    if bins > sequence_length:
        raise ValueError("bins cannot exceed sequence length")
    boundaries = [index * sequence_length // bins for index in range(bins + 1)]
    return tuple(zip(boundaries[:-1], boundaries[1:]))


def summarize_sums(sums, counts, ranges):
    """Convert position-bin accumulators into JSON-safe records."""

    records = []
    for (start, end), total, count in zip(ranges, sums, counts):
        count = int(count)
        if count:
            records.append(
                {
                    "start": start,
                    "end": end,
                    "tokens": count,
                    "loss": float(total) / count,
                }
            )
    return records


@torch.no_grad()
def evaluate_position_loss(
    model, loader, steps, source_ids, sequence_length, bins, trailing_tokens
):
    """Aggregate unreduced causal losses without retaining vocabulary-sized logits."""

    ranges = position_ranges(sequence_length, bins)
    trailing_tokens = min(positive_integer(trailing_tokens, "trailing tokens"), sequence_length)
    device = next(model.parameters()).device
    source_indices = {source_id: index for index, source_id in enumerate(source_ids)}
    bin_sums = torch.zeros(len(ranges), device=device, dtype=torch.float64)
    bin_counts = torch.zeros(len(ranges), device=device, dtype=torch.int64)
    source_bin_sums = torch.zeros(len(source_ids), len(ranges), device=device, dtype=torch.float64)
    source_bin_counts = torch.zeros(len(source_ids), len(ranges), device=device, dtype=torch.int64)
    trailing_sum = torch.zeros((), device=device, dtype=torch.float64)
    trailing_count = torch.zeros((), device=device, dtype=torch.int64)
    source_trailing_sums = torch.zeros(len(source_ids), device=device, dtype=torch.float64)
    source_trailing_counts = torch.zeros(len(source_ids), device=device, dtype=torch.int64)

    model.eval()
    for _ in range(steps):
        inputs, targets, state = next(loader)
        losses = model(inputs, targets, loss_reduction="none").view(inputs.shape)
        if not torch.isfinite(losses).all():
            raise FloatingPointError("position loss evaluation produced non-finite losses")
        source_index = source_indices[state["selected_source"]]
        for bin_index, (start, end) in enumerate(ranges):
            values = losses[:, start:end]
            total = values.double().sum()
            count = values.numel()
            bin_sums[bin_index] += total
            bin_counts[bin_index] += count
            source_bin_sums[source_index, bin_index] += total
            source_bin_counts[source_index, bin_index] += count
        values = losses[:, -trailing_tokens:]
        total = values.double().sum()
        count = values.numel()
        trailing_sum += total
        trailing_count += count
        source_trailing_sums[source_index] += total
        source_trailing_counts[source_index] += count

    bin_sums = bin_sums.cpu().tolist()
    bin_counts = bin_counts.cpu().tolist()
    source_bin_sums = source_bin_sums.cpu().tolist()
    source_bin_counts = source_bin_counts.cpu().tolist()
    source_trailing_sums = source_trailing_sums.cpu().tolist()
    source_trailing_counts = source_trailing_counts.cpu().tolist()
    total_sum = sum(bin_sums)
    total_count = sum(bin_counts)
    return {
        "loss": total_sum / total_count,
        "position_bins": summarize_sums(bin_sums, bin_counts, ranges),
        "trailing_loss": float(trailing_sum.cpu()) / int(trailing_count.cpu()),
        "trailing_tokens_per_sequence": trailing_tokens,
        "source_position_bins": {
            source_id: summarize_sums(source_bin_sums[index], source_bin_counts[index], ranges)
            for index, source_id in enumerate(source_ids)
            if sum(source_bin_counts[index])
        },
        "source_trailing_loss": {
            source_id: source_trailing_sums[index] / source_trailing_counts[index]
            for index, source_id in enumerate(source_ids)
            if source_trailing_counts[index]
        },
    }


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
    sequence_length = positive_integer(
        args.sequence_length or train["sequence_length"], "sequence length"
    )
    eval_tokens = positive_integer(args.eval_tokens, "evaluation tokens")
    batch_size = positive_integer(args.batch_size or train["device_batch_size"], "batch size")
    bins = positive_integer(args.bins, "bins")
    trailing_tokens = positive_integer(args.trailing_tokens, "trailing tokens")
    loss_backend = args.loss_backend or train.get("loss_backend", "torch")
    device = torch.device(args.device)
    model, metadata = load_checkpoint_model(checkpoint_dir, step, device, loss_backend=loss_backend)
    checkpoint_rope_scaling_factor = model.config.rope_scaling_factor
    if args.rope_scaling_factor is not None:
        if not math.isfinite(args.rope_scaling_factor) or args.rope_scaling_factor <= 0:
            raise ValueError("RoPE scaling factor must be positive and finite")
        for rotary in model.rotary.values():
            rotary.scaling_factor = args.rope_scaling_factor
        model.config = replace(model.config, rope_scaling_factor=args.rope_scaling_factor)
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
    metrics = evaluate_position_loss(
        eval_model,
        loader,
        steps,
        tuple(source["id"] for source in manifest["sources"]),
        sequence_length,
        bins,
        trailing_tokens,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    duration = time.perf_counter() - started
    report = {
        "format": "speck_position_loss_evaluation",
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
        "bins": bins,
        "loss_backend": loss_backend,
        "compiled": not args.no_compile,
        "seconds": duration,
        "tokens_per_second": evaluated_tokens / duration,
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "device": str(device),
        "torch_version": torch.__version__,
        "checkpoint_rope_scaling_factor": checkpoint_rope_scaling_factor,
        "evaluation_rope_scaling_factor": model.config.rope_scaling_factor,
        **metrics,
    }
    output = (
        args.output
        or Path(base_dir())
        / "evaluations"
        / "position-loss"
        / train["run"]
        / f"{Path(data_experiment).name}-{sequence_length}-{step}.json"
    )
    atomic_json(output, report)
    return report


def main(argv=None):
    report = run(arguments(argv))
    print(
        f"loss {report['loss']:.5f} | trailing {report['trailing_loss']:.5f} | "
        f"{report['evaluated_tokens']:,} tokens"
    )


if __name__ == "__main__":
    main()
