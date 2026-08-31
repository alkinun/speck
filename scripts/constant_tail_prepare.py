"""Prepare a constant-LR branch experiment from a completed base checkpoint."""

import argparse
import json
import os
import shutil
from pathlib import Path

from scripts.base_train import changed_branch_settings
from speck.architecture import ArchitectureConfig
from speck.checkpoint import load_metadata
from speck.config import load_experiment
from speck.dataloader import manifest_fingerprint
from speck.dataset import load_manifest, resolve_data_dir
from speck.train import lr_scale

_EXPERIMENT_FILES = ("data.json", "model.json", "tokenizer.json")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent_experiment", type=Path)
    parser.add_argument("output_experiment", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--train-tokens", type=int, required=True)
    parser.add_argument(
        "--run", default=None, help="branch run name; defaults to output directory name"
    )
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


def constant_tail_config(parent, metadata, train_tokens, run):
    if isinstance(train_tokens, bool) or not isinstance(train_tokens, int) or train_tokens < 1:
        raise ValueError("constant-tail train_tokens must be a positive integer")
    if not isinstance(run, str) or not run:
        raise ValueError("constant-tail run must be a non-empty string")
    resolved = metadata["resolved"]
    train = {
        **parent,
        "checkpoint_tokens": [],
        "device_batch_size": resolved["device_batch_size"],
        "loss_backend": resolved.get("loss_backend", "torch"),
        "lr": next_learning_rate(metadata),
        "lr_schedule": "constant",
        "min_lr": 1.0,
        "output_dir": None,
        "run": run,
        "train_tokens": train_tokens,
        "warmup_steps": 0,
    }
    train.pop("decay_steps", None)
    changed = changed_branch_settings(
        resolved,
        {**train, "world_size": resolved["world_size"]},
        allow_schedule_change=True,
    )
    if changed:
        raise ValueError(f"parent experiment drifted from checkpoint: {', '.join(changed)}")
    return train


def prepare(parent_experiment, output_experiment, checkpoint_dir, step, train_tokens, run=None):
    parent_experiment = Path(parent_experiment).expanduser().resolve()
    output_experiment = Path(output_experiment).expanduser().resolve()
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    if output_experiment.exists():
        raise FileExistsError(f"branch experiment already exists: {output_experiment}")
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

    run = run or output_experiment.name
    train = constant_tail_config(configs["train"], metadata, train_tokens, run)
    building = output_experiment.with_name(output_experiment.name + ".building")
    shutil.rmtree(building, ignore_errors=True)
    output_experiment.parent.mkdir(parents=True, exist_ok=True)
    building.mkdir()
    try:
        for filename in _EXPERIMENT_FILES:
            shutil.copy2(parent_experiment / filename, building / filename)
        (building / "train.json").write_text(
            json.dumps(train, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(building, output_experiment)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return train, metadata["resolved"]["world_size"]


def main():
    args = parse_args()
    train, world_size = prepare(
        args.parent_experiment,
        args.output_experiment,
        args.checkpoint_dir,
        args.step,
        args.train_tokens,
        args.run,
    )
    print(
        f"Prepared {args.output_experiment} at constant LR {train['lr']:.12g}; "
        f"launch with world size {world_size} and --branch-schedule new"
    )


if __name__ == "__main__":
    main()
