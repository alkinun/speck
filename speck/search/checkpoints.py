"""resumable content-addressed checkpoints for quality trajectories."""

import os
import random
import tempfile
from dataclasses import asdict, dataclass

import numpy as np
import torch

from speck.search.artifacts import ArtifactRecord, ArtifactStore


checkpoint_format_version = 1


@dataclass(frozen=True)
class RunCheckpoint:
    artifact: ArtifactRecord
    architecture_digest: str
    protocol_digest: str
    seed_bundle_digest: str
    steps: int
    tokens: int
    parent_digest: str | None = None
    format_version: int = checkpoint_format_version

    def __post_init__(self):
        digests = (
            self.architecture_digest,
            self.protocol_digest,
            self.seed_bundle_digest,
        )
        if any(not digest for digest in digests):
            raise ValueError("checkpoint identity digests cannot be empty")
        if self.steps < 0 or self.tokens < 0:
            raise ValueError("checkpoint progress cannot be negative")
        if self.parent_digest is not None and len(self.parent_digest) != 64:
            raise ValueError("checkpoint parent digests must be sha256 values")

    def export(self):
        return asdict(self)


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    cuda = state.get("cuda", [])
    if cuda:
        if not torch.cuda.is_available() or len(cuda) != torch.cuda.device_count():
            raise ValueError("checkpoint cuda rng state does not match this worker")
        torch.cuda.set_rng_state_all(cuda)


def save_run_checkpoint(
    store,
    *,
    architecture_digest,
    protocol_digest,
    seed_bundle_digest,
    steps,
    tokens,
    model_state,
    optimizer_state,
    data_state,
    parent=None,
    extra=None,
):
    if not isinstance(store, ArtifactStore):
        raise TypeError("checkpoint store must be an artifact store")
    parent_digest = parent.artifact.digest if parent is not None else None
    identity = {
        "architecture_digest": architecture_digest,
        "format_version": checkpoint_format_version,
        "parent_digest": parent_digest,
        "protocol_digest": protocol_digest,
        "seed_bundle_digest": seed_bundle_digest,
        "steps": steps,
        "tokens": tokens,
    }
    payload = {
        "identity": identity,
        "model": model_state,
        "optimizer": optimizer_state,
        "data_state": data_state,
        "rng": capture_rng_state(),
        "extra": extra or {},
    }
    descriptor, temporary = tempfile.mkstemp(
        dir=store.root,
        prefix="checkpoint-",
        suffix=".pt",
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        artifact = store.put_file(
            "quality_checkpoint",
            temporary,
            "application/x-pytorch",
        )
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return RunCheckpoint(artifact=artifact, **identity)


def load_run_checkpoint(
    store,
    checkpoint,
    *,
    device="cpu",
    architecture_digest=None,
    protocol_digest=None,
    seed_bundle_digest=None,
    restore_rng=False,
):
    if not isinstance(store, ArtifactStore):
        raise TypeError("checkpoint store must be an artifact store")
    store.verify(checkpoint.artifact)
    payload = torch.load(
        store.path(checkpoint.artifact),
        map_location=device,
        weights_only=False,
    )
    identity = payload.get("identity", {})
    expected = {
        "architecture_digest": architecture_digest,
        "protocol_digest": protocol_digest,
        "seed_bundle_digest": seed_bundle_digest,
    }
    for name, value in expected.items():
        if value is not None and identity.get(name) != value:
            raise ValueError(f"checkpoint {name} does not match")
    stored = RunCheckpoint(artifact=checkpoint.artifact, **identity)
    if stored != checkpoint:
        raise ValueError("checkpoint manifest does not match its payload")
    if restore_rng:
        restore_rng_state(payload["rng"])
    return payload
