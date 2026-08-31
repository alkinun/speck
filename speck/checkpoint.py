"""Provide atomic training checkpoints."""

import hashlib
import json
import os
import re
from pathlib import Path

import torch


def _validate_step(step):
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("checkpoint step must be a non-negative integer")
    return step


def save(directory, step, model, optimizer, metadata, timing=None):
    step = _validate_step(step)
    os.makedirs(directory, exist_ok=True)
    paths = {
        "model": os.path.join(directory, f"model_{step:06d}.pt"),
        "optimizer": os.path.join(directory, f"optimizer_{step:06d}.pt"),
        "metadata": os.path.join(directory, f"metadata_{step:06d}.json"),
        "timing": os.path.join(directory, f"timing_{step:06d}.json"),
        "complete": os.path.join(directory, f"complete_{step:06d}"),
    }
    for path in paths.values():
        if os.path.exists(path):
            os.remove(path)
    torch.save(model, paths["model"] + ".tmp")
    torch.save(optimizer, paths["optimizer"] + ".tmp")
    with open(paths["metadata"] + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    timing = timing() if callable(timing) else timing
    if timing is not None:
        with open(paths["timing"] + ".tmp", "w", encoding="utf-8") as handle:
            json.dump(timing, handle, indent=2)
    os.replace(paths["model"] + ".tmp", paths["model"])
    os.replace(paths["optimizer"] + ".tmp", paths["optimizer"])
    os.replace(paths["metadata"] + ".tmp", paths["metadata"])
    if timing is not None:
        os.replace(paths["timing"] + ".tmp", paths["timing"])
    with open(paths["complete"] + ".tmp", "w", encoding="utf-8") as handle:
        handle.write("complete\n")
    os.replace(paths["complete"] + ".tmp", paths["complete"])


def completed_steps(directory):
    if not os.path.isdir(directory):
        return []
    steps = []
    for name in os.listdir(directory):
        match = re.fullmatch(r"complete_(\d+)", name)
        if match is None:
            continue
        step = int(match.group(1))
        if name == f"complete_{step:06d}":
            steps.append(step)
    return sorted(steps)


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
            f"timing_{step:06d}.json",
        )
        for name in names:
            path = os.path.join(directory, name)
            if os.path.exists(path):
                os.remove(path)


def load(directory, step, device):
    step = _validate_step(step)
    if not os.path.exists(os.path.join(directory, f"complete_{step:06d}")):
        raise FileNotFoundError(f"checkpoint {step} is incomplete")
    model = torch.load(os.path.join(directory, f"model_{step:06d}.pt"), map_location=device)
    optimizer = torch.load(os.path.join(directory, f"optimizer_{step:06d}.pt"), map_location=device)
    metadata = load_metadata(directory, step)
    return model, optimizer, metadata


def load_model(directory, step, device):
    step = _validate_step(step)
    if not os.path.exists(os.path.join(directory, f"complete_{step:06d}")):
        raise FileNotFoundError(f"checkpoint {step} is incomplete")
    path = os.path.join(directory, f"model_{step:06d}.pt")
    return torch.load(path, map_location=device)


def load_metadata(directory, step):
    step = _validate_step(step)
    if not os.path.exists(os.path.join(directory, f"complete_{step:06d}")):
        raise FileNotFoundError(f"checkpoint {step} is incomplete")
    path = os.path.join(directory, f"metadata_{step:06d}.json")
    with open(path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("step") != step:
        raise ValueError(f"checkpoint metadata step does not match {step}")
    return metadata


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_identity(directory):
    """Return a stable identity for every file in a directory tree."""

    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"directory does not exist: {directory}")
    files = sorted(candidate for candidate in directory.rglob("*") if candidate.is_file())
    if not files:
        raise ValueError(f"directory is empty: {directory}")
    digest = hashlib.sha256()
    entries = []
    for candidate in files:
        relative = candidate.relative_to(directory).as_posix()
        checksum = file_sha256(candidate)
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(checksum.encode())
        digest.update(b"\n")
        entries.append({"path": relative, "sha256": checksum, "bytes": candidate.stat().st_size})
    return {"path": str(directory), "sha256": digest.hexdigest(), "files": entries}


def checkpoint_identity(directory, step):
    """Return stable model and metadata identities for a completed checkpoint."""

    step = _validate_step(step)
    load_metadata(directory, step)
    directory = Path(directory).expanduser().resolve()
    model = directory / f"model_{step:06d}.pt"
    optimizer = directory / f"optimizer_{step:06d}.pt"
    metadata = directory / f"metadata_{step:06d}.json"
    return {
        "directory": str(directory),
        "step": step,
        "model_sha256": file_sha256(model),
        "optimizer_sha256": file_sha256(optimizer),
        "metadata_sha256": file_sha256(metadata),
    }


def save_timing(directory, step, timing):
    """Atomically record post-checkpoint timing that cannot be known before completion."""

    step = _validate_step(step)
    if not os.path.exists(os.path.join(directory, f"complete_{step:06d}")):
        raise FileNotFoundError(f"checkpoint {step} is incomplete")
    path = Path(directory) / f"timing_{step:06d}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(timing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_timing(directory, step):
    """Load optional post-checkpoint timing, falling back to metadata for old checkpoints."""

    step = _validate_step(step)
    path = Path(directory) / f"timing_{step:06d}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
