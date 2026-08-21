import random

import numpy as np
import pytest
import torch

from speck.search.artifacts import ArtifactStore
from speck.search.checkpoints import RunCheckpoint, load_run_checkpoint, save_run_checkpoint


def optimization_step(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    inputs = torch.randn(3, 2)
    targets = torch.randn(3, 2)
    loss = torch.nn.functional.mse_loss(model(inputs), targets)
    loss.backward()
    optimizer.step()


def test_run_checkpoint_restores_training_and_rng_state(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    checkpoint = save_run_checkpoint(
        store,
        architecture_digest="architecture",
        protocol_digest="protocol",
        seed_bundle_digest="seeds",
        steps=1,
        tokens=32,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        data_state={"offset": 32},
    )
    expected_random = (random.random(), np.random.random(), torch.rand(1))
    payload = load_run_checkpoint(
        store,
        checkpoint,
        architecture_digest="architecture",
        protocol_digest="protocol",
        seed_bundle_digest="seeds",
        restore_rng=True,
    )
    actual_random = (random.random(), np.random.random(), torch.rand(1))
    assert expected_random[0] == actual_random[0]
    assert expected_random[1] == actual_random[1]
    assert torch.equal(expected_random[2], actual_random[2])
    assert payload["data_state"] == {"offset": 32}


def test_run_checkpoint_records_parent_lineage(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    values = {
        "architecture_digest": "architecture",
        "protocol_digest": "protocol",
        "seed_bundle_digest": "seeds",
        "model_state": {},
        "optimizer_state": {},
        "data_state": {},
    }
    parent = save_run_checkpoint(store, steps=1, tokens=32, **values)
    child = save_run_checkpoint(
        store,
        steps=2,
        tokens=64,
        parent=parent,
        **values,
    )
    assert child.parent_digest == parent.artifact.digest
    assert child.artifact.digest != parent.artifact.digest
    assert RunCheckpoint.from_dict(child.export()) == child


def test_run_checkpoint_rejects_incompatible_parent(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    parent = save_run_checkpoint(
        store,
        architecture_digest="architecture",
        protocol_digest="protocol",
        seed_bundle_digest="seeds",
        steps=1,
        tokens=32,
        model_state={},
        optimizer_state={},
        data_state={},
    )
    with pytest.raises(ValueError, match="parent identity"):
        save_run_checkpoint(
            store,
            architecture_digest="different",
            protocol_digest="protocol",
            seed_bundle_digest="seeds",
            steps=2,
            tokens=64,
            model_state={},
            optimizer_state={},
            data_state={},
            parent=parent,
        )


def test_run_checkpoint_rejects_protocol_mismatch(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    checkpoint = save_run_checkpoint(
        store,
        architecture_digest="architecture",
        protocol_digest="protocol",
        seed_bundle_digest="seeds",
        steps=1,
        tokens=32,
        model_state={},
        optimizer_state={},
        data_state={},
    )
    with pytest.raises(ValueError, match="protocol_digest"):
        load_run_checkpoint(store, checkpoint, protocol_digest="different")


def test_run_checkpoint_matches_uninterrupted_training(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    torch.manual_seed(9)
    uninterrupted = torch.nn.Linear(2, 2)
    uninterrupted_optimizer = torch.optim.AdamW(uninterrupted.parameters(), lr=0.01)
    optimization_step(uninterrupted, uninterrupted_optimizer)
    checkpoint = save_run_checkpoint(
        store,
        architecture_digest="architecture",
        protocol_digest="protocol",
        seed_bundle_digest="seeds",
        steps=1,
        tokens=32,
        model_state=uninterrupted.state_dict(),
        optimizer_state=uninterrupted_optimizer.state_dict(),
        data_state={"offset": 32},
    )
    optimization_step(uninterrupted, uninterrupted_optimizer)

    resumed = torch.nn.Linear(2, 2)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.01)
    payload = load_run_checkpoint(store, checkpoint, restore_rng=True)
    resumed.load_state_dict(payload["model"])
    resumed_optimizer.load_state_dict(payload["optimizer"])
    optimization_step(resumed, resumed_optimizer)
    for expected, actual in zip(uninterrupted.parameters(), resumed.parameters()):
        assert torch.equal(expected, actual)
