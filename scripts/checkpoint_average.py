"""Average completed checkpoints from one training trajectory."""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch

from speck.checkpoint import checkpoint_identity, load_metadata, load_model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _trajectory(metadata):
    resolved = metadata["resolved"]
    return {
        "config": metadata["config"],
        "manifest": metadata["manifest"],
        "training_phase": metadata.get("training_phase"),
        "run": resolved.get("run"),
        "parent_checkpoint": resolved.get("parent_checkpoint"),
        "schedule_step_offset": resolved.get("schedule_step_offset", 0),
        "schedule_steps": resolved.get("schedule_steps", resolved["steps"]),
    }


def _add_state(total, state, reference):
    if set(state) != set(reference):
        raise ValueError("checkpoint model keys do not match")
    for name, tensor in state.items():
        expected = reference[name]
        if tensor.shape != expected.shape or tensor.dtype != expected.dtype:
            raise ValueError(f"checkpoint tensor does not match: {name}")
        if tensor.is_floating_point():
            total[name].add_(tensor.float())
        elif not torch.equal(tensor, expected):
            raise ValueError(f"non-floating checkpoint tensor changed: {name}")


def average_checkpoints(checkpoint_dir, steps):
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    if len(steps) < 2 or steps != sorted(set(steps)):
        raise ValueError("average requires at least two sorted unique checkpoint steps")

    metadatas = [load_metadata(checkpoint_dir, step) for step in steps]
    trajectory = _trajectory(metadatas[0])
    positions = []
    for step, metadata in zip(steps, metadatas):
        if metadata["step"] != step or _trajectory(metadata) != trajectory:
            raise ValueError("checkpoints do not belong to one training trajectory")
        positions.append(metadata["global_tokens"])
    if positions != sorted(set(positions)):
        raise ValueError("checkpoint global token positions must increase")

    reference = load_model(checkpoint_dir, steps[0], "cpu")
    total = {
        name: tensor.float().clone() if tensor.is_floating_point() else tensor.clone()
        for name, tensor in reference.items()
    }
    for step in steps[1:]:
        _add_state(total, load_model(checkpoint_dir, step, "cpu"), reference)
    count = len(steps)
    state = {
        name: (
            tensor.div_(count).to(reference[name].dtype) if tensor.is_floating_point() else tensor
        )
        for name, tensor in total.items()
    }
    metadata = {
        "format": "speck_model_average",
        "format_version": 1,
        "average": {
            "accumulation_dtype": "float32",
            "count": count,
            "weight": 1 / count,
        },
        "config": metadatas[-1]["config"],
        "manifest": metadatas[-1]["manifest"],
        "resolved": metadatas[-1]["resolved"],
        "global_tokens": positions[-1],
        "checkpoints": [checkpoint_identity(checkpoint_dir, step) for step in steps],
    }
    return state, metadata


def write_average(output_dir, state, metadata, force=False):
    output_dir = Path(output_dir).expanduser().resolve()
    building = output_dir.with_name(output_dir.name + ".building")
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"average already exists: {output_dir}")
        shutil.rmtree(output_dir)
    shutil.rmtree(building, ignore_errors=True)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    building.mkdir()
    try:
        torch.save(state, building / "model.pt")
        (building / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (building / "complete").write_text("complete\n", encoding="utf-8")
        os.replace(building, output_dir)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise


def main():
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if checkpoint_dir == output_dir:
        raise ValueError("average output must differ from the checkpoint directory")
    state, metadata = average_checkpoints(checkpoint_dir, args.steps)
    write_average(output_dir, state, metadata, args.force)
    print(f"Averaged {len(args.steps)} checkpoints into {output_dir}")


if __name__ == "__main__":
    main()
