"""Provide atomic training checkpoints."""

import json
import os
import re

import torch


def save(directory, step, model, optimizer, metadata):
    os.makedirs(directory, exist_ok=True)
    paths = {
        "model": os.path.join(directory, f"model_{step:06d}.pt"),
        "optimizer": os.path.join(directory, f"optimizer_{step:06d}.pt"),
        "metadata": os.path.join(directory, f"metadata_{step:06d}.json"),
        "complete": os.path.join(directory, f"complete_{step:06d}"),
    }
    for path in paths.values():
        if os.path.exists(path):
            os.remove(path)
    torch.save(model, paths["model"] + ".tmp")
    torch.save(optimizer, paths["optimizer"] + ".tmp")
    with open(paths["metadata"] + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    os.replace(paths["model"] + ".tmp", paths["model"])
    os.replace(paths["optimizer"] + ".tmp", paths["optimizer"])
    os.replace(paths["metadata"] + ".tmp", paths["metadata"])
    with open(paths["complete"] + ".tmp", "w", encoding="utf-8") as handle:
        handle.write("complete\n")
    os.replace(paths["complete"] + ".tmp", paths["complete"])


def completed_steps(directory):
    if not os.path.isdir(directory):
        return []
    return sorted(
        int(match.group(1))
        for name in os.listdir(directory)
        if (match := re.fullmatch(r"complete_(\d+)", name))
    )


def latest(directory):
    steps = completed_steps(directory)
    return steps[-1] if steps else None


def prune(directory, keep):
    """Keep only the newest completed checkpoints in one run directory."""

    if not isinstance(keep, int) or keep < 1:
        raise ValueError("checkpoint retention must be a positive integer")
    for step in completed_steps(directory)[:-keep]:
        names = (
            f"complete_{step:06d}",
            f"model_{step:06d}.pt",
            f"optimizer_{step:06d}.pt",
            f"metadata_{step:06d}.json",
        )
        for name in names:
            path = os.path.join(directory, name)
            if os.path.exists(path):
                os.remove(path)


def load(directory, step, device):
    if not os.path.exists(os.path.join(directory, f"complete_{step:06d}")):
        raise FileNotFoundError(f"checkpoint {step} is incomplete")
    model = torch.load(os.path.join(directory, f"model_{step:06d}.pt"), map_location=device)
    optimizer = torch.load(os.path.join(directory, f"optimizer_{step:06d}.pt"), map_location=device)
    with open(os.path.join(directory, f"metadata_{step:06d}.json"), encoding="utf-8") as handle:
        metadata = json.load(handle)
    return model, optimizer, metadata


def load_model(directory, step, device):
    if not os.path.exists(os.path.join(directory, f"complete_{step:06d}")):
        raise FileNotFoundError(f"checkpoint {step} is incomplete")
    path = os.path.join(directory, f"model_{step:06d}.pt")
    return torch.load(path, map_location=device)
