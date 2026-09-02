"""Prepare a progressive-context experiment bound to an exact parent checkpoint."""

import argparse
import json
import os
import shutil
from pathlib import Path

from scripts.base_train import context_compatible_architecture
from speck.architecture import ArchitectureConfig
from speck.checkpoint import checkpoint_identity, load_metadata
from speck.config import load_experiment


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent_experiment", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--train-tokens", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--rope-theta", type=float, default=None)
    parser.add_argument("--rope-scaling-factor", type=float, default=1.0)
    parser.add_argument("--loss-backend", choices=("torch", "liger"), default=None)
    parser.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--min-lr", type=float, default=0.1)
    parser.add_argument("--data-experiment", type=Path, default=None)
    parser.add_argument("--run", default=None)
    return parser.parse_args(argv)


def stage_configs(
    parent_configs,
    metadata,
    *,
    sequence_length,
    train_tokens,
    lr,
    rope_theta=None,
    rope_scaling_factor=1.0,
    loss_backend=None,
    activation_checkpointing=None,
    warmup_steps=100,
    min_lr=0.1,
    run,
):
    integer_values = {
        "sequence length": sequence_length,
        "train tokens": train_tokens,
        "warmup steps": warmup_steps,
    }
    for name, value in integer_values.items():
        minimum = 0 if name == "warmup steps" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}")
    if loss_backend is not None and loss_backend not in {"torch", "liger"}:
        raise ValueError("context stage loss backend must be torch or liger")
    if activation_checkpointing is not None and not isinstance(activation_checkpointing, bool):
        raise ValueError("context stage activation checkpointing must be boolean or null")
    if sequence_length > parent_configs["model"].get("max_position_embeddings", 4_096):
        maximum = sequence_length
    else:
        maximum = parent_configs["model"].get("max_position_embeddings", 4_096)
    model = {
        **parent_configs["model"],
        "max_position_embeddings": maximum,
        "rope_scaling_factor": rope_scaling_factor,
    }
    if rope_theta is not None:
        model["rope_theta"] = rope_theta
    if not context_compatible_architecture(metadata["config"], model):
        raise ValueError("context stage changed the parent parameter topology")
    if not isinstance(run, str) or not run:
        raise ValueError("context stage run must be a non-empty string")
    train = {
        **parent_configs["train"],
        "activation_checkpointing": (
            parent_configs["train"].get("activation_checkpointing", False)
            if activation_checkpointing is None
            else activation_checkpointing
        ),
        "checkpoint_tokens": [],
        "device_batch_size": 1,
        "lr": lr,
        "lr_schedule": "cosine",
        "loss_backend": loss_backend or parent_configs["train"].get("loss_backend", "torch"),
        "min_lr": min_lr,
        "output_dir": None,
        "run": run,
        "sequence_length": sequence_length,
        "train_tokens": train_tokens,
        "training_phase": "context_extension",
        "warmup_steps": warmup_steps,
    }
    return model, train


def prepare(args):
    parent = args.parent_experiment.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"context stage already exists: {output}")
    configs = load_experiment(parent, "data", "long_context", "model", "tokenizer", "train")
    metadata = load_metadata(checkpoint_dir, args.step)
    if (
        ArchitectureConfig.from_dict(configs["model"]).settings()
        != ArchitectureConfig.from_dict(metadata["config"]).settings()
    ):
        raise ValueError("parent experiment architecture does not match checkpoint")
    data_experiment = (args.data_experiment or parent).expanduser().resolve()
    data = load_experiment(data_experiment, "data", "tokenizer")
    if data["tokenizer"] != configs["tokenizer"]:
        raise ValueError("context data tokenizer does not match the parent")
    run = args.run or output.name
    model, train = stage_configs(
        configs,
        metadata,
        sequence_length=args.sequence_length,
        train_tokens=args.train_tokens,
        lr=args.lr,
        rope_theta=args.rope_theta,
        rope_scaling_factor=args.rope_scaling_factor,
        loss_backend=args.loss_backend,
        activation_checkpointing=args.activation_checkpointing,
        warmup_steps=args.warmup_steps,
        min_lr=args.min_lr,
        run=run,
    )
    contract = {
        "format": "speck_context_stage",
        "format_version": 1,
        "parent_checkpoint": checkpoint_identity(checkpoint_dir, args.step),
        "parent_experiment": str(parent),
        "data_experiment": str(data_experiment),
        "sequence_length": args.sequence_length,
        "train_tokens": args.train_tokens,
        "rope_theta": model.get("rope_theta"),
        "rope_scaling_factor": model["rope_scaling_factor"],
        "loss_backend": train["loss_backend"],
        "activation_checkpointing": train["activation_checkpointing"],
    }
    building = output.with_name(output.name + ".building")
    shutil.rmtree(building, ignore_errors=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    building.mkdir()
    try:
        values = {
            "data.json": data["data"],
            "long_context.json": configs["long_context"],
            "model.json": model,
            "tokenizer.json": configs["tokenizer"],
            "train.json": train,
            "context_stage.json": contract,
        }
        for filename, value in values.items():
            (building / filename).write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        os.replace(building, output)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return contract


def main():
    args = arguments()
    contract = prepare(args)
    print(
        f"Prepared {args.output_dir} at {contract['sequence_length']:,} tokens; launch with "
        f"--branch-from {args.checkpoint_dir} --branch-step {args.step} "
        "--branch-kind context --branch-schedule new"
    )


if __name__ == "__main__":
    main()
