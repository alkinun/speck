"""Prepare matched inherited-schedule and constant-LR tail experiments."""

import argparse
import json
import os
import shutil
from pathlib import Path

from scripts.base_train import changed_branch_settings
from speck.architecture import ArchitectureConfig
from speck.checkpoint import checkpoint_identity, load_metadata
from speck.config import load_experiment
from speck.dataloader import manifest_fingerprint
from speck.dataset import load_manifest, resolve_data_dir
from speck.train import lr_scale

_EXPERIMENT_FILES = ("data.json", "model.json", "tokenizer.json")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent_experiment", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--train-tokens", type=int, required=True)
    parser.add_argument("--run-prefix", default=None, help="defaults to output directory name")
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    return parser.parse_args(argv)


def next_learning_rate(metadata):
    resolved = metadata["resolved"]
    schedule_step = resolved.get("schedule_step_offset", 0) + metadata["step"]
    schedule_steps = resolved.get("schedule_steps", resolved["steps"])
    if schedule_step >= schedule_steps:
        raise ValueError("parent learning-rate schedule is exhausted")
    scale = lr_scale(
        schedule_step,
        schedule_steps,
        resolved["warmup_steps"],
        resolved["min_lr"],
        resolved.get("lr_schedule", "cosine"),
        resolved.get("decay_steps"),
    )
    return resolved["lr"] * scale


def tail_configs(
    parent,
    metadata,
    train_tokens,
    run_prefix,
    save_every=None,
    eval_every=None,
):
    if isinstance(train_tokens, bool) or not isinstance(train_tokens, int) or train_tokens < 1:
        raise ValueError("tail train_tokens must be a positive integer")
    if not isinstance(run_prefix, str) or not run_prefix:
        raise ValueError("tail run prefix must be a non-empty string")
    for name, value in (("save_every", save_every), ("eval_every", eval_every)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError(f"tail {name} must be a non-negative integer")

    resolved = metadata["resolved"]
    tail_steps = (train_tokens + resolved["batch_tokens"] - 1) // resolved["batch_tokens"]
    schedule_step = resolved.get("schedule_step_offset", 0) + metadata["step"]
    schedule_steps = resolved.get("schedule_steps", resolved["steps"])
    if schedule_step + tail_steps > schedule_steps:
        raise ValueError("tail exceeds the remaining parent learning-rate schedule")
    control = {
        **parent,
        "checkpoint_tokens": [],
        "device_batch_size": resolved["device_batch_size"],
        "loss_backend": resolved.get("loss_backend", "torch"),
        "lr": resolved["lr"],
        "lr_schedule": resolved.get("lr_schedule", "cosine"),
        "min_lr": resolved["min_lr"],
        "output_dir": None,
        "run": f"{run_prefix}-Control",
        "train_tokens": train_tokens,
        "warmup_steps": resolved["warmup_steps"],
    }
    decay_steps = resolved.get("decay_steps")
    if decay_steps is None:
        control.pop("decay_steps", None)
    else:
        control["decay_steps"] = decay_steps
    if save_every is not None:
        control["save_every"] = save_every
    if eval_every is not None:
        control["eval_every"] = eval_every

    constant = {
        **control,
        "lr": next_learning_rate(metadata),
        "lr_schedule": "constant",
        "min_lr": 1.0,
        "run": f"{run_prefix}-Constant",
        "warmup_steps": 0,
    }
    constant.pop("decay_steps", None)
    world_size = resolved["world_size"]
    control_changes = changed_branch_settings(resolved, {**control, "world_size": world_size})
    constant_changes = changed_branch_settings(
        resolved,
        {**constant, "world_size": world_size},
        allow_schedule_change=True,
    )
    if control_changes or constant_changes:
        changed = sorted(set(control_changes + constant_changes))
        raise ValueError(f"parent experiment drifted from checkpoint: {', '.join(changed)}")
    return control, constant


def _write_experiment(parent, output, train):
    output.mkdir()
    for filename in _EXPERIMENT_FILES:
        shutil.copy2(parent / filename, output / filename)
    (output / "train.json").write_text(
        json.dumps(train, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def prepare(
    parent_experiment,
    output_dir,
    checkpoint_dir,
    step,
    train_tokens,
    run_prefix=None,
    save_every=None,
    eval_every=None,
):
    parent_experiment = Path(parent_experiment).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"tail pair already exists: {output_dir}")
    configs = load_experiment(parent_experiment, "data", "model", "tokenizer", "train")
    metadata = load_metadata(checkpoint_dir, step)
    architecture = ArchitectureConfig.from_dict(configs["model"]).settings()
    if architecture != ArchitectureConfig.from_dict(metadata["config"]).settings():
        raise ValueError("parent experiment architecture does not match checkpoint")
    data_dir = resolve_data_dir(
        configs["data"].get("output_dir"), configs["data"].get("output_name")
    )
    if manifest_fingerprint(load_manifest(data_dir)) != metadata["manifest"]:
        raise ValueError("parent experiment packed data does not match checkpoint")

    run_prefix = run_prefix or output_dir.name
    control, constant = tail_configs(
        configs["train"],
        metadata,
        train_tokens,
        run_prefix,
        save_every,
        eval_every,
    )
    consumed_tokens = (
        (train_tokens + control["batch_tokens"] - 1) // control["batch_tokens"]
    ) * control["batch_tokens"]
    pair = {
        "format": "speck_tail_pair",
        "format_version": 1,
        "parent_checkpoint": checkpoint_identity(checkpoint_dir, step),
        "parent_global_tokens": metadata["global_tokens"],
        "manifest": metadata["manifest"],
        "train_tokens": train_tokens,
        "consumed_tokens": consumed_tokens,
        "world_size": metadata["resolved"]["world_size"],
        "save_every": control["save_every"],
        "eval_every": control["eval_every"],
        "control": {
            "experiment": "control",
            "run": control["run"],
            "schedule": "inherit",
        },
        "constant": {
            "experiment": "constant",
            "run": constant["run"],
            "schedule": "new",
            "lr": constant["lr"],
        },
    }
    building = output_dir.with_name(output_dir.name + ".building")
    shutil.rmtree(building, ignore_errors=True)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    building.mkdir()
    try:
        _write_experiment(parent_experiment, building / "control", control)
        _write_experiment(parent_experiment, building / "constant", constant)
        (building / "pair.json").write_text(
            json.dumps(pair, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(building, output_dir)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return pair


def main():
    args = parse_args()
    pair = prepare(
        args.parent_experiment,
        args.output_dir,
        args.checkpoint_dir,
        args.step,
        args.train_tokens,
        args.run_prefix,
        args.save_every,
        args.eval_every,
    )
    print(
        f"Prepared {args.output_dir} at constant LR {pair['constant']['lr']:.12g}; "
        f"launch both arms with world size {pair['world_size']}"
    )


if __name__ == "__main__":
    main()
