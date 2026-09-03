"""Prepare a progressive-context experiment bound to an exact parent checkpoint."""

import argparse
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

from scripts.base_train import context_compatible_architecture
from speck.architecture import (
    ArchitectureConfig,
    AttentionSpec,
    BlockConfig,
    BlockGroup,
    StageConfig,
)
from speck.checkpoint import checkpoint_identity, load_metadata
from speck.config import load_experiment
from speck.dataloader import manifest_fingerprint
from speck.dataset import load_manifest, resolve_data_dir


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
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument(
        "--global-attention-layers",
        type=int,
        nargs="*",
        default=None,
        help="logical sliding-attention layers to promote to global attention",
    )
    parser.add_argument(
        "--global-attention-rope-dim",
        type=int,
        default=None,
        help="override RoPE dimensions only on promoted global layers; use 0 for NoPE",
    )
    return parser.parse_args(argv)


def promote_global_attention_layers(model, layer_indices, rope_dim=None):
    """Promote selected logical sliding-attention layers without changing parameters."""

    if any(
        not isinstance(index, int) or isinstance(index, bool) or index < 0
        for index in layer_indices
    ) or len(set(layer_indices)) != len(layer_indices):
        raise ValueError("global attention layer indices must be unique non-negative integers")
    if rope_dim is not None and (
        not isinstance(rope_dim, int) or isinstance(rope_dim, bool) or rope_dim < 0
    ):
        raise ValueError("global attention RoPE dimensions must be a non-negative integer")
    config = ArchitectureConfig.from_dict(model)
    if any(group.repeat != 1 for group in config.blocks):
        raise ValueError("attention-scope promotion requires materialized logical layers")
    requested = set(layer_indices)
    promoted = set()
    groups = []
    for layer_index, group in enumerate(config.blocks):
        stages = []
        for stage in group.block.stages:
            branches = []
            for branch in stage.branches:
                if isinstance(branch, AttentionSpec) and layer_index in requested:
                    if branch.scope != "sliding":
                        raise ValueError("attention-scope promotion requires sliding parent layers")
                    changes = {"scope": "global", "window_size": None}
                    if rope_dim is not None:
                        changes["rope_dim"] = rope_dim
                    branch = replace(branch, **changes)
                    promoted.add(layer_index)
                branches.append(branch)
            stages.append(StageConfig(tuple(branches)))
        groups.append(
            BlockGroup(
                BlockConfig(group.block.hidden_size, tuple(stages)),
                repeat=group.repeat,
                weight_sharing=group.weight_sharing,
            )
        )
    if promoted != requested:
        missing = ", ".join(map(str, sorted(requested - promoted)))
        raise ValueError(f"requested global layers are not sliding-attention layers: {missing}")
    return replace(config, blocks=tuple(groups)).export()


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
    wandb_group=None,
    global_attention_layers=None,
    global_attention_rope_dim=None,
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
    global_attention_layers = tuple(global_attention_layers or ())
    if global_attention_rope_dim is not None and not global_attention_layers:
        raise ValueError("global attention RoPE override requires promoted global layers")
    if global_attention_layers:
        model = promote_global_attention_layers(
            model,
            global_attention_layers,
            rope_dim=global_attention_rope_dim,
        )
    if not context_compatible_architecture(
        metadata["config"],
        model,
        allow_attention_scope_change=bool(global_attention_layers),
    ):
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
        "allow_attention_scope_change": bool(global_attention_layers),
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
    if wandb_group is not None:
        if not isinstance(wandb_group, str) or not wandb_group:
            raise ValueError("context stage W&B group must be a non-empty string")
        train["wandb_group"] = wandb_group
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
    data_dir = resolve_data_dir(data["data"].get("output_dir"), data["data"].get("output_name"))
    data_manifest = load_manifest(data_dir)
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
        wandb_group=args.wandb_group,
        global_attention_layers=args.global_attention_layers,
        global_attention_rope_dim=args.global_attention_rope_dim,
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
        "data_directory": str(data_dir.resolve()),
        "data_manifest": manifest_fingerprint(data_manifest),
        "sequence_length": args.sequence_length,
        "train_tokens": args.train_tokens,
        "rope_theta": model.get("rope_theta"),
        "rope_scaling_factor": model["rope_scaling_factor"],
        "loss_backend": train["loss_backend"],
        "activation_checkpointing": train["activation_checkpointing"],
        "global_attention_layers": list(args.global_attention_layers or ()),
        "global_attention_rope_dim": args.global_attention_rope_dim,
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
