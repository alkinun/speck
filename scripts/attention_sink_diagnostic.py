"""Measure attention-sink behavior from sampled late-query attention rows."""

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

from scripts.infer import load_checkpoint_model
from speck.checkpoint import checkpoint_identity, latest
from speck.common import base_dir
from speck.config import load_experiment
from speck.dataloader import manifest_fingerprint, packed_loader
from speck.dataset import load_manifest, resolve_data_dir
from speck.model import Attention, rotate
from speck.tokenizer import get_tokenizer


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_experiment", type=Path)
    parser.add_argument("--data-experiment", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--queries", type=int, default=32)
    parser.add_argument("--minimum-query-fraction", type=float, default=0.5)
    parser.add_argument("--prefix-tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def query_indices(sequence_length, queries, minimum_fraction):
    sequence_length = positive_integer(sequence_length, "sequence length")
    queries = positive_integer(queries, "queries")
    if not math.isfinite(minimum_fraction) or not 0 <= minimum_fraction < 1:
        raise ValueError("minimum query fraction must be in [0, 1)")
    first = max(1, math.floor((sequence_length - 1) * minimum_fraction))
    available = sequence_length - first
    queries = min(queries, available)
    if queries == 1:
        return (sequence_length - 1,)
    return tuple(
        first + index * (sequence_length - 1 - first) // (queries - 1) for index in range(queries)
    )


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class SinkAccumulator:
    def __init__(self, prefix_tokens):
        self.prefix_tokens = positive_integer(prefix_tokens, "prefix tokens")
        self.rows = 0
        self.first_mass = 0.0
        self.first_enrichment = 0.0
        self.prefix_mass = 0.0
        self.prefix_enrichment = 0.0
        self.recent_mass = 0.0
        self.maximum_mass = 0.0
        self.normalized_entropy = 0.0
        self.argmax_first = 0
        self.query_positions = []

    def update(self, probabilities, query_position):
        probabilities = probabilities.detach().float()
        keys = probabilities.size(-1)
        rows = probabilities.numel() // keys
        prefix = min(self.prefix_tokens, keys)
        first = probabilities[..., 0]
        prefix_mass = probabilities[..., :prefix].sum(dim=-1)
        recent_mass = probabilities[..., -prefix:].sum(dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(dim=-1)
        normalized_entropy = entropy / math.log(keys) if keys > 1 else torch.ones_like(entropy)
        self.rows += rows
        self.first_mass += first.sum().item()
        self.first_enrichment += (first * keys).sum().item()
        self.prefix_mass += prefix_mass.sum().item()
        self.prefix_enrichment += (prefix_mass * keys / prefix).sum().item()
        self.recent_mass += recent_mass.sum().item()
        self.maximum_mass += probabilities.max(dim=-1).values.sum().item()
        self.normalized_entropy += normalized_entropy.sum().item()
        self.argmax_first += int((probabilities.argmax(dim=-1) == 0).sum().item())
        self.query_positions.append(query_position)

    def merge(self, other):
        self.rows += other.rows
        self.first_mass += other.first_mass
        self.first_enrichment += other.first_enrichment
        self.prefix_mass += other.prefix_mass
        self.prefix_enrichment += other.prefix_enrichment
        self.recent_mass += other.recent_mass
        self.maximum_mass += other.maximum_mass
        self.normalized_entropy += other.normalized_entropy
        self.argmax_first += other.argmax_first
        self.query_positions.extend(other.query_positions)

    def report(self):
        if not self.rows:
            raise ValueError("cannot report an empty sink accumulator")
        return {
            "rows": self.rows,
            "first_token_mass": self.first_mass / self.rows,
            "first_token_enrichment_over_uniform": self.first_enrichment / self.rows,
            "first_token_argmax_fraction": self.argmax_first / self.rows,
            "prefix_tokens": self.prefix_tokens,
            "prefix_mass": self.prefix_mass / self.rows,
            "prefix_enrichment_over_uniform": self.prefix_enrichment / self.rows,
            "recent_prefix_sized_mass": self.recent_mass / self.rows,
            "maximum_token_mass": self.maximum_mass / self.rows,
            "normalized_entropy": self.normalized_entropy / self.rows,
            "minimum_query_position": min(self.query_positions),
            "maximum_query_position": max(self.query_positions),
        }


def attention_qk(module, x, rotary, position):
    batch, length, _ = x.shape
    query = (
        module.q_proj(x).view(batch, length, module.q_heads, module.spec.head_dim).transpose(1, 2)
    )
    key = (
        module.k_proj(x)
        .view(batch, length, module.spec.num_key_value_heads, module.spec.head_dim)
        .transpose(1, 2)
    )
    query, key = module.q_norm(query), module.k_norm(key)
    rotary_dim = module.spec.head_dim if module.spec.rope_dim is None else module.spec.rope_dim
    if rotary_dim:
        cosine, sine = rotary(position, length, query.dtype)
        query = rotate(query, cosine, sine, rotary_dim)
        key = rotate(key, cosine, sine, rotary_dim)
    if module.q_heads != module.spec.num_key_value_heads:
        repetitions = module.q_heads // module.spec.num_key_value_heads
        key = key.repeat_interleave(repetitions, dim=1)
    return query, key


@torch.inference_mode()
def collect(model, loader, batches, queries, minimum_fraction, prefix_tokens):
    accumulators = {}
    handles = []
    for name, module in model.named_modules():
        if not isinstance(module, Attention) or module.spec.scope != "global":
            continue
        accumulator = SinkAccumulator(prefix_tokens)
        accumulators[name] = accumulator

        def hook(current, inputs, _name=name):
            x, rotary, position = inputs[:3]
            query, key = attention_qk(current, x, rotary, position)
            scale = current.spec.head_dim**-0.5
            for query_index in query_indices(query.size(2), queries, minimum_fraction):
                scores = torch.einsum(
                    "bhd,bhkd->bhk",
                    query[:, :, query_index].float(),
                    key[:, :, : query_index + 1].float(),
                )
                accumulators[_name].update((scores * scale).softmax(dim=-1), query_index)

        handles.append(module.register_forward_pre_hook(hook))
    if not handles:
        raise ValueError("checkpoint has no global attention modules")
    model.eval()
    try:
        for _ in range(batches):
            inputs, _, _ = next(loader)
            model(inputs, last_token_only=True)
    finally:
        for handle in handles:
            handle.remove()
    combined = SinkAccumulator(prefix_tokens)
    for accumulator in accumulators.values():
        combined.merge(accumulator)
    return {
        "aggregate": combined.report(),
        "modules": {name: accumulator.report() for name, accumulator in accumulators.items()},
    }


def run(args):
    model_configs = load_experiment(args.model_experiment, "tokenizer", "train")
    data_experiment = args.data_experiment or args.model_experiment
    data_configs = load_experiment(data_experiment, "data", "tokenizer")
    if data_configs["tokenizer"] != model_configs["tokenizer"]:
        raise ValueError("diagnostic data tokenizer does not match the model")
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
    batch_size = positive_integer(args.batch_size, "batch size")
    batches = positive_integer(args.batches, "batches")
    queries = positive_integer(args.queries, "queries")
    prefix_tokens = positive_integer(args.prefix_tokens, "prefix tokens")
    device = torch.device(args.device)
    model, metadata = load_checkpoint_model(checkpoint_dir, step, device)
    if sequence_length > model.config.max_position_embeddings:
        raise ValueError("diagnostic sequence exceeds the model context")
    tokenizer = get_tokenizer(**data_configs["tokenizer"])
    data_dir = resolve_data_dir(
        data_configs["data"].get("output_dir"), data_configs["data"].get("output_name")
    )
    manifest = load_manifest(data_dir)
    loader = packed_loader(
        tokenizer,
        batch_size,
        sequence_length,
        "val",
        device=device,
        data_dir=data_dir,
    )
    statistics = collect(
        model,
        loader,
        batches,
        queries,
        args.minimum_query_fraction,
        prefix_tokens,
    )
    report = {
        "format": "speck_attention_sink_diagnostic",
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_experiment": str(args.model_experiment.expanduser().resolve()),
        "checkpoint": checkpoint_identity(checkpoint_dir, step),
        "checkpoint_global_tokens": metadata.get("global_tokens"),
        "data_experiment": str(data_experiment.expanduser().resolve()),
        "data_directory": str(data_dir.expanduser().resolve()),
        "data_manifest": manifest_fingerprint(manifest),
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "batches": batches,
        "evaluated_tokens": batch_size * sequence_length * batches,
        "queries_per_layer_per_sequence": min(queries, sequence_length - 1),
        "minimum_query_fraction": args.minimum_query_fraction,
        "prefix_tokens": prefix_tokens,
        "device": str(device),
        "torch_version": torch.__version__,
        **statistics,
    }
    output = (
        args.output
        or Path(base_dir())
        / "evaluations"
        / "attention-sink"
        / train["run"]
        / f"{step}-{sequence_length}.json"
    )
    atomic_json(output, report)
    return report


def main(argv=None):
    report = run(arguments(argv))
    aggregate = report["aggregate"]
    print(
        f"first mass {aggregate['first_token_mass']:.6f} | "
        f"enrichment {aggregate['first_token_enrichment_over_uniform']:.2f}x | "
        f"argmax {aggregate['first_token_argmax_fraction']:.2%}"
    )


if __name__ == "__main__":
    main()
