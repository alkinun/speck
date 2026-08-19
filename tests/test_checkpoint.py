import json

import torch

from speck.checkpoint import latest, load, save


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

    torch.save({"weight": torch.tensor([2.0])}, tmp_path / "model_000004.pt")
    (tmp_path / "metadata_000004.json").write_text(json.dumps({"step": 4}))
    assert latest(tmp_path) == 3
    try:
        load(tmp_path, 4, "cpu")
    except FileNotFoundError as error:
        assert "incomplete" in str(error)
    else:
        raise AssertionError("incomplete checkpoint was accepted")
