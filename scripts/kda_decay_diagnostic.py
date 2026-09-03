"""Measure trained KDA log-decay distributions on packed data."""

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.infer import load_checkpoint_model
from speck.checkpoint import checkpoint_identity, latest
from speck.common import base_dir
from speck.config import load_experiment
from speck.dataloader import manifest_fingerprint, packed_loader
from speck.dataset import load_manifest, resolve_data_dir
from speck.model import KimiDeltaAttention
from speck.tokenizer import get_tokenizer

THRESHOLDS = (-5.0, -10.0, -20.0, -40.0, -80.0)
QUANTILES = (0.0, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_experiment", type=Path)
    parser.add_argument("--data-experiment", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--sample-values", type=int, default=262_144)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class DecayAccumulator:
    def __init__(self, sample_values):
        self.sample_values = positive_integer(sample_values, "sample values")
        self.count = 0
        self.total = 0.0
        self.square_total = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.below = {threshold: 0 for threshold in THRESHOLDS}
        self.samples = []

    def update(self, values):
        values = values.detach().float().flatten()
        self.count += values.numel()
        self.total += values.sum(dtype=torch.float64).item()
        self.square_total += values.square().sum(dtype=torch.float64).item()
        self.minimum = min(self.minimum, values.min().item())
        self.maximum = max(self.maximum, values.max().item())
        for threshold in THRESHOLDS:
            self.below[threshold] += int((values < threshold).sum().item())
        stride = max(1, math.ceil(values.numel() / self.sample_values))
        self.samples.append(values[::stride][: self.sample_values].cpu())

    def merge(self, other):
        self.count += other.count
        self.total += other.total
        self.square_total += other.square_total
        self.minimum = min(self.minimum, other.minimum)
        self.maximum = max(self.maximum, other.maximum)
        for threshold in THRESHOLDS:
            self.below[threshold] += other.below[threshold]
        self.samples.extend(other.samples)

    def report(self):
        if not self.count:
            raise ValueError("cannot report an empty decay accumulator")
        sample = torch.cat(self.samples)
        if sample.numel() > self.sample_values:
            stride = math.ceil(sample.numel() / self.sample_values)
            sample = sample[::stride][: self.sample_values]
        quantiles = torch.quantile(sample, torch.tensor(QUANTILES)).tolist()
        mean = self.total / self.count
        variance = max(0.0, self.square_total / self.count - mean * mean)
        return {
            "count": self.count,
            "mean": mean,
            "population_stddev": math.sqrt(variance),
            "minimum": self.minimum,
            "maximum": self.maximum,
            "fractions_below": {
                str(int(threshold)): self.below[threshold] / self.count for threshold in THRESHOLDS
            },
            "quantiles": {str(quantile): value for quantile, value in zip(QUANTILES, quantiles)},
            "sample_count": sample.numel(),
        }


def kda_log_decay(module, x):
    batch, length, _ = x.shape
    logits = module.decay_up_projection(module.decay_down_projection(x)).view(
        batch,
        length,
        module.spec.num_value_heads,
        module.spec.key_head_dim,
    )
    bias = module.decay_bias.view(module.spec.num_value_heads, module.spec.key_head_dim)
    return -module.log_rates.float().exp()[None, None, :, None] * F.softplus(logits.float() + bias)


@torch.inference_mode()
def collect(model, loader, batches, sample_values):
    accumulators = {}
    handles = []
    for name, module in model.named_modules():
        if not isinstance(module, KimiDeltaAttention):
            continue
        accumulator = DecayAccumulator(sample_values)
        accumulators[name] = accumulator

        def hook(current, inputs, _name=name):
            accumulators[_name].update(kda_log_decay(current, inputs[0]))

        handles.append(module.register_forward_pre_hook(hook))
    if not handles:
        raise ValueError("checkpoint has no KDA modules")
    model.eval()
    try:
        for _ in range(batches):
            inputs, _, _ = next(loader)
            model(inputs, last_token_only=True)
    finally:
        for handle in handles:
            handle.remove()
    combined = DecayAccumulator(sample_values)
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
    sample_values = positive_integer(args.sample_values, "sample values")
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
    statistics = collect(model, loader, batches, sample_values)
    report = {
        "format": "speck_kda_decay_diagnostic",
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
        "sample_values_per_scope": sample_values,
        "k3_candidate_log_decay_floor": -5.0,
        "device": str(device),
        "torch_version": torch.__version__,
        **statistics,
    }
    output = (
        args.output
        or Path(base_dir())
        / "evaluations"
        / "kda-decay"
        / train["run"]
        / f"{step}-{sequence_length}.json"
    )
    atomic_json(output, report)
    return report


def main(argv=None):
    report = run(arguments(argv))
    aggregate = report["aggregate"]
    print(
        f"g mean {aggregate['mean']:.4f} | min {aggregate['minimum']:.4f} | "
        f"below -5 {aggregate['fractions_below']['-5']:.4%}"
    )


if __name__ == "__main__":
    main()
