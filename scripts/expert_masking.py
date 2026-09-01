"""Measure held-out LM-loss sensitivity to masking each routed layer."""

import argparse
import json
import os
from pathlib import Path

import torch

from speck.architecture import ArchitectureConfig
from speck.checkpoint import checkpoint_identity, latest, load_metadata, load_model
from speck.config import load_experiment
from speck.dataloader import manifest_fingerprint, packed_loader
from speck.dataset import load_manifest, resolve_data_dir, verify_shards
from speck.model import build_model
from speck.tokenizer import get_tokenizer


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--step", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--tokens", type=int)
    args = parser.parse_args(argv)
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.tokens is not None and args.tokens < 1:
        parser.error("--tokens must be positive")
    return args


@torch.no_grad()
def replay_loss(model, batches, device, masked_layer=None):
    loss_sum = torch.zeros((), device=device)
    target_count = 0
    masks = () if masked_layer is None else (masked_layer,)
    for inputs, targets in batches:
        inputs = inputs.to(device)
        targets = targets.to(device)
        loss_sum += model(
            inputs,
            targets,
            loss_reduction="sum",
            masked_routed_layers=masks,
        )
        target_count += int((targets != -100).sum())
    if target_count == 0:
        raise ValueError("masking replay contains no scored targets")
    return (loss_sum / target_count).item()


@torch.no_grad()
def evaluate_masking(model, batches, device):
    """Replay exactly the same materialized batches for baseline and every mask."""

    layers = tuple(model.routed_operations())
    if not layers:
        raise ValueError("expert masking requires at least one routed layer")
    model.eval()
    baseline = replay_loss(model, batches, device)
    results = []
    for layer in layers:
        loss = replay_loss(model, batches, device, layer)
        results.append({"layer": layer, "lm_loss": loss, "loss_delta": loss - baseline})
    return baseline, results


def materialize_validation_batches(
    tokenizer,
    data_dir,
    batch_size,
    sequence_length,
    requested_tokens,
    available_tokens,
):
    tokens_per_batch = batch_size * sequence_length
    batches = min(requested_tokens, available_tokens) // tokens_per_batch
    if batches < 1:
        raise ValueError("requested masking replay is smaller than one validation batch")
    loader = packed_loader(
        tokenizer,
        batch_size,
        sequence_length,
        "val",
        device="cpu",
        data_dir=data_dir,
    )
    replay = []
    for _ in range(batches):
        inputs, targets, _ = next(loader)
        replay.append((inputs.clone(), targets.clone()))
    return tuple(replay), batches * tokens_per_batch


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(args):
    experiment = args.experiment.expanduser().resolve()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    step = args.step if args.step is not None else latest(checkpoint_dir)
    if step is None:
        raise FileNotFoundError(f"no completed checkpoint in {checkpoint_dir}")
    metadata = load_metadata(checkpoint_dir, step)
    configs = load_experiment(experiment, "data", "tokenizer", "model", "train")
    tokenizer = get_tokenizer(**configs["tokenizer"])
    data_dir = resolve_data_dir(
        configs["data"].get("output_dir"), configs["data"].get("output_name")
    )
    manifest = load_manifest(data_dir)
    verify_shards(data_dir, manifest)
    manifest_hash = manifest_fingerprint(manifest)
    if metadata["manifest"] != manifest_hash:
        raise ValueError("checkpoint and validation dataset manifests do not match")
    device = torch.device(args.device)
    model = build_model(
        configs["model"],
        tokenizer.vocab_size,
        tokenizer.bos_id,
        tokenizer.eos_id,
    ).to(device)
    stored = ArchitectureConfig.from_dict(metadata["config"])
    if stored.settings() != model.config.settings():
        raise ValueError("checkpoint and masking architecture do not match")
    model.load_state_dict(load_model(checkpoint_dir, step, device))
    batch_size = args.batch_size or configs["train"]["device_batch_size"]
    requested_tokens = args.tokens or configs["train"]["final_eval_tokens"]
    batches, evaluated_tokens = materialize_validation_batches(
        tokenizer,
        data_dir,
        batch_size,
        configs["train"]["sequence_length"],
        requested_tokens,
        manifest["splits"]["val"]["tokens"],
    )
    baseline, results = evaluate_masking(model, batches, device)
    report = {
        "format": "speck_routed_expert_masking",
        "format_version": 1,
        "checkpoint": checkpoint_identity(checkpoint_dir, step),
        "checkpoint_global_tokens": metadata.get("global_tokens"),
        "architecture": {
            "digest": model.config.digest,
            "settings": model.config.settings(),
            "total_parameters": model.parameter_count(),
            "active_parameters": model.active_parameter_count(),
        },
        "dataset": {
            "directory": str(Path(data_dir).expanduser().resolve()),
            "manifest_sha256": manifest_hash,
            "format": manifest["format"],
            "split": "val",
        },
        "replay": {
            "batch_size": batch_size,
            "batches": len(batches),
            "sequence_length": configs["train"]["sequence_length"],
            "tokens": evaluated_tokens,
        },
        "baseline_lm_loss": baseline,
        "layers": results,
    }
    atomic_json(args.output.expanduser().resolve(), report)
    return report


def main(argv=None):
    args = arguments(argv)
    report = run(args)
    print(
        f"Wrote {len(report['layers'])} routed-layer sensitivities to "
        f"{args.output.expanduser().resolve()}"
    )


if __name__ == "__main__":
    main()
