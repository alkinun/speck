import json
from pathlib import Path

import pytest
import torch

import speck.checkpoint as checkpoint
from speck.checkpoint import (
    checkpoint_identity,
    completed_steps,
    latest,
    load,
    load_metadata,
    load_model,
    load_timing,
    prune,
    save,
)


def test_checkpoint_is_visible_only_after_completion(tmp_path):
    save(
        tmp_path,
        3,
        {"weight": torch.tensor([1.0])},
        {"state": "optimizer"},
        {"step": 3},
    )
    assert latest(tmp_path) == 3
    model, optimizer, metadata = load(tmp_path, 3, "cpu")
    assert model["weight"].item() == 1.0
    assert optimizer == {"state": "optimizer"}
    assert metadata == {"step": 3}
    assert load_model(tmp_path, 3, "cpu")["weight"].item() == 1.0
    assert load_metadata(tmp_path, 3) == {"step": 3}
    identity = checkpoint_identity(tmp_path, 3)
    assert identity["step"] == 3
    assert {
        len(identity["model_sha256"]),
        len(identity["optimizer_sha256"]),
        len(identity["metadata_sha256"]),
    } == {64}

    torch.save({"weight": torch.tensor([2.0])}, tmp_path / "model_000004.pt")
    (tmp_path / "metadata_000004.json").write_text(json.dumps({"step": 4}))
    assert latest(tmp_path) == 3
    try:
        load(tmp_path, 4, "cpu")
    except FileNotFoundError as error:
        assert "incomplete" in str(error)
    else:
        raise AssertionError("incomplete checkpoint was accepted")


def test_checkpoint_retention_removes_only_old_complete_steps(tmp_path):
    for step in (1, 2, 3):
        save(tmp_path, step, {"step": step}, {}, {"step": step})
    (tmp_path / "model_000004.pt").write_bytes(b"incomplete")

    prune(tmp_path, keep=2)

    assert completed_steps(tmp_path) == [2, 3]
    assert not (tmp_path / "model_000001.pt").exists()
    assert (tmp_path / "model_000004.pt").read_bytes() == b"incomplete"


def test_checkpoint_discovery_ignores_noncanonical_markers(tmp_path):
    for name in (
        "complete_000000",
        "complete_000003",
        "complete_1000000",
        "complete_3",
        "complete_0000003",
        "complete_00000003",
        "complete_-1",
        "complete_invalid",
    ):
        (tmp_path / name).touch()

    assert completed_steps(tmp_path) == [0, 3, 1_000_000]
    assert latest(tmp_path) == 1_000_000


def test_checkpoint_metadata_must_match_its_filename_step(tmp_path):
    save(tmp_path, 3, {}, {}, {"step": 2})
    with pytest.raises(ValueError, match="metadata step"):
        load_metadata(tmp_path, 3)
    with pytest.raises(ValueError, match="metadata step"):
        load(tmp_path, 3, "cpu")


def test_checkpoint_timing_sidecar_is_optional_and_pruned(tmp_path):
    save(tmp_path, 1, {}, {}, {"step": 1})
    assert load_timing(tmp_path, 1) is None
    save(tmp_path, 2, {}, {}, {"step": 2}, timing={"active_seconds": 2.5})
    assert load_timing(tmp_path, 2) == {"active_seconds": 2.5}
    save(tmp_path, 3, {}, {}, {"step": 3})

    prune(tmp_path, keep=1)

    assert not (tmp_path / "timing_000002.json").exists()


def test_checkpoint_can_publish_timing_before_completion(tmp_path):
    save(
        tmp_path,
        3,
        {},
        {},
        {"step": 3},
        timing=lambda: {"checkpoint_seconds": 1.25},
    )
    assert load_timing(tmp_path, 3) == {"checkpoint_seconds": 1.25}


@pytest.mark.parametrize("failed_call", (1, 2))
def test_failed_tensor_serialization_preserves_completed_predecessor(
    tmp_path, monkeypatch, failed_call
):
    save(tmp_path, 1, {"weight": torch.tensor([1.0])}, {"old": True}, {"step": 1})
    original_save = checkpoint.torch.save
    calls = 0

    def fail_once(value, path):
        nonlocal calls
        calls += 1
        if calls == failed_call:
            raise OSError("injected serialization failure")
        return original_save(value, path)

    monkeypatch.setattr(checkpoint.torch, "save", fail_once)
    with pytest.raises(OSError, match="injected serialization failure"):
        save(tmp_path, 1, {"weight": torch.tensor([2.0])}, {"old": False}, {"step": 1})

    model, optimizer, metadata = load(tmp_path, 1, "cpu")
    assert model["weight"].item() == 1.0
    assert optimizer == {"old": True}
    assert metadata == {"step": 1}
    assert latest(tmp_path) == 1
    assert not any(".tmp." in path.name or ".backup." in path.name for path in tmp_path.iterdir())


def test_failed_metadata_or_timing_preserves_completed_predecessor(tmp_path):
    save(
        tmp_path,
        1,
        {"weight": torch.tensor([1.0])},
        {"old": True},
        {"step": 1},
        timing={"active_seconds": 1.0},
    )
    with pytest.raises(TypeError):
        save(tmp_path, 1, {}, {}, {"step": 1, "invalid": {1}})
    assert load_model(tmp_path, 1, "cpu")["weight"].item() == 1.0
    assert load_timing(tmp_path, 1) == {"active_seconds": 1.0}

    def failed_timing():
        raise RuntimeError("injected timing failure")

    with pytest.raises(RuntimeError, match="injected timing failure"):
        save(tmp_path, 1, {}, {}, {"step": 1}, timing=failed_timing)
    assert load_model(tmp_path, 1, "cpu")["weight"].item() == 1.0
    assert load_timing(tmp_path, 1) == {"active_seconds": 1.0}


def test_partial_publication_failure_rolls_back_every_predecessor_file(tmp_path, monkeypatch):
    save(
        tmp_path,
        1,
        {"weight": torch.tensor([1.0])},
        {"old": True},
        {"step": 1},
        timing={"active_seconds": 1.0},
    )
    original_replace = checkpoint.os.replace
    injected = False

    def fail_new_optimizer(source, destination):
        nonlocal injected
        if (
            not injected
            and Path(destination).name == "optimizer_000001.pt"
            and ".tmp." in Path(source).name
        ):
            injected = True
            raise OSError("injected publication failure")
        return original_replace(source, destination)

    monkeypatch.setattr(checkpoint.os, "replace", fail_new_optimizer)
    with pytest.raises(OSError, match="injected publication failure"):
        save(
            tmp_path,
            1,
            {"weight": torch.tensor([2.0])},
            {"old": False},
            {"step": 1},
            timing={"active_seconds": 2.0},
        )

    model, optimizer, metadata = load(tmp_path, 1, "cpu")
    assert model["weight"].item() == 1.0
    assert optimizer == {"old": True}
    assert metadata == {"step": 1}
    assert load_timing(tmp_path, 1) == {"active_seconds": 1.0}
    assert not any(".tmp." in path.name or ".backup." in path.name for path in tmp_path.iterdir())


def test_discovery_recovers_predecessor_after_uncatchable_publication_interrupt(
    tmp_path, monkeypatch
):
    save(
        tmp_path,
        1,
        {"weight": torch.tensor([1.0])},
        {"old": True},
        {"step": 1},
        timing={"active_seconds": 1.0},
    )
    original_replace = checkpoint.os.replace
    injected = False

    def interrupt_new_optimizer(source, destination):
        nonlocal injected
        if (
            not injected
            and Path(destination).name == "optimizer_000001.pt"
            and ".tmp." in Path(source).name
        ):
            injected = True
            raise KeyboardInterrupt
        return original_replace(source, destination)

    monkeypatch.setattr(checkpoint.os, "replace", interrupt_new_optimizer)
    with pytest.raises(KeyboardInterrupt):
        save(
            tmp_path,
            1,
            {"weight": torch.tensor([2.0])},
            {"old": False},
            {"step": 1},
            timing={"active_seconds": 2.0},
        )
    assert any(path.name.startswith(".checkpoint-transaction-") for path in tmp_path.iterdir())

    monkeypatch.setattr(checkpoint.os, "replace", original_replace)
    assert latest(tmp_path) == 1
    model, optimizer, metadata = load(tmp_path, 1, "cpu")
    assert model["weight"].item() == 1.0
    assert optimizer == {"old": True}
    assert metadata == {"step": 1}
    assert load_timing(tmp_path, 1) == {"active_seconds": 1.0}
    assert not any(
        ".tmp." in path.name
        or ".backup." in path.name
        or path.name.startswith(".checkpoint-transaction-")
        for path in tmp_path.iterdir()
    )


def test_successful_same_step_replacement_removes_stale_optional_timing(tmp_path):
    save(tmp_path, 1, {}, {}, {"step": 1}, timing={"active_seconds": 1.0})
    save(tmp_path, 1, {"new": True}, {"new": True}, {"step": 1})

    model, optimizer, metadata = load(tmp_path, 1, "cpu")
    assert model == {"new": True}
    assert optimizer == {"new": True}
    assert metadata == {"step": 1}
    assert load_timing(tmp_path, 1) is None


@pytest.mark.parametrize("step", (-1, 1.5, True, "3"))
def test_checkpoint_operations_reject_invalid_steps(tmp_path, step):
    directory = tmp_path / "checkpoints"

    with pytest.raises(ValueError, match="non-negative integer"):
        save(directory, step, {}, {}, {})
    with pytest.raises(ValueError, match="non-negative integer"):
        load(directory, step, "cpu")
    with pytest.raises(ValueError, match="non-negative integer"):
        load_model(directory, step, "cpu")

    assert not directory.exists()
