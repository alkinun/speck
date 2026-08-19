"""atomic training checkpoints."""

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


def latest(directory):
    if not os.path.isdir(directory):
        return None
    steps = [int(match.group(1)) for name in os.listdir(directory) if (match := re.fullmatch(r"complete_(\d+)", name))]
    return max(steps) if steps else None


def load(directory, step, device):
    if not os.path.exists(os.path.join(directory, f"complete_{step:06d}")):
        raise FileNotFoundError(f"checkpoint {step} is incomplete")
    model = torch.load(os.path.join(directory, f"model_{step:06d}.pt"), map_location=device)
    optimizer = torch.load(os.path.join(directory, f"optimizer_{step:06d}.pt"), map_location=device)
    with open(os.path.join(directory, f"metadata_{step:06d}.json"), encoding="utf-8") as handle:
        metadata = json.load(handle)
    return model, optimizer, metadata
