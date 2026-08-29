import json

import pytest
import torch

from speck.checkpoint import completed_steps, latest, load, load_model, prune, save


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
