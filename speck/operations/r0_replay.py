"""Persist and verify the state needed for a fresh-process R0 next-step comparison."""

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from speck.provenance.io import durable_json, file_sha256
from speck.training import checkpoint


def capture_rng(device):
    name, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        "python": random.getstate(),
        "numpy": (name, keys.tolist(), position, has_gauss, cached_gaussian),
    }


def restore_rng(value, device):
    torch.set_rng_state(value["torch_cpu"])
    if device.type == "cuda":
        if value["torch_cuda"] is None:
            raise ValueError("CUDA restart is missing its RNG state")
        torch.cuda.set_rng_state(value["torch_cuda"], device)
    elif value["torch_cuda"] is not None:
        raise ValueError("checkpoint RNG device differs")
    random.setstate(value["python"])
    name, keys, position, has_gauss, cached_gaussian = value["numpy"]
    np.random.set_state(
        (name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached_gaussian)
    )


def rng_probe(device):
    """Exercise every persisted generator even when the model itself has no stochastic layers."""
    digest = hashlib.sha256()
    digest.update(torch.randint(2**30, (32,)).numpy().tobytes())
    digest.update(np.random.randint(2**30, size=32, dtype=np.int64).tobytes())
    digest.update(json.dumps([random.random() for _ in range(32)]).encode())
    if device.type == "cuda":
        digest.update(torch.randint(2**30, (32,), device=device).cpu().numpy().tobytes())
    return digest.hexdigest()


def save_rng(path, device):
    path = Path(path)
    # A unique attempt owns this file. It is not restartable until the later manifest is durable.
    if path.exists():
        raise FileExistsError("preserve previous RNG payload")
    with path.open("xb") as handle:
        torch.save(capture_rng(device), handle)
        handle.flush()
        os.fsync(handle.fileno())
    return {"path": path.name, "sha256": file_sha256(path)}


def publish_reference(
    directory, count, model, optimizer, metadata, baseline_identity, rng_identity, loss, device
):
    directory = Path(directory)
    expected_directory = directory / "expected"
    expected_metadata = {
        **metadata,
        "step": count + 1,
        "next_microbatch_ordinal": metadata["next_microbatch_ordinal"]
        + metadata["next_microbatch_ordinal"] // count,
        "rng_probe_sha256": rng_probe(device),
    }
    checkpoint.save(
        expected_directory, count + 1, model.state_dict(), optimizer.state_dict(), expected_metadata
    )
    result = {
        "format": "speck_r0_fresh_process_reference",
        "format_version": 1,
        "request_sha256": metadata["request_sha256"],
        "rank": metadata["rank"],
        "world_size": metadata["world_size"],
        "baseline": baseline_identity,
        "rng": rng_identity,
        "expected": checkpoint.checkpoint_identity(expected_directory, count + 1),
        "next_step_loss": loss,
        "rng_probe_sha256": expected_metadata["rng_probe_sha256"],
        "producer_pid": os.getpid(),
    }
    durable_json(directory / "restart-reference.json", result)
    return result


def verified_reference(directory, request, rank, world_size):
    directory = Path(directory).resolve()
    reference = json.loads((directory / "restart-reference.json").read_text())
    if (
        reference.get("format"),
        reference.get("format_version"),
        reference["request_sha256"],
        reference["rank"],
        reference["world_size"],
    ) != ("speck_r0_fresh_process_reference", 1, request["request_sha256"], rank, world_size):
        raise ValueError("fresh-process reference identity differs")
    producer = json.loads((directory.parent / f"rank-{rank}-result.json").read_text())
    if reference["producer_pid"] == os.getpid():
        raise ValueError("fresh-process replay requires a different worker process")
    if (
        producer["status"] != "restart_reference_ready"
        or producer["request_sha256"] != request["request_sha256"]
        or producer["restart_reference"] != reference
    ):
        raise ValueError("restart reference differs from completed producer result")
    count = request["settings"]["warmup_steps"] + request["settings"]["measured_steps"]
    if reference["baseline"]["step"] != count or reference["expected"]["step"] != count + 1:
        raise ValueError("restart reference step differs")
    for role in ("baseline", "expected"):
        binding = reference[role]
        path = Path(binding["directory"]).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("restart checkpoint is outside its reference directory")
        if checkpoint.checkpoint_identity(path, binding["step"]) != binding:
            raise ValueError("restart checkpoint payload identity differs")
    rng_path = directory / reference["rng"]["path"]
    if (
        rng_path.resolve().parent != directory
        or file_sha256(rng_path) != reference["rng"]["sha256"]
    ):
        raise ValueError("restart RNG payload identity differs")
    return reference
