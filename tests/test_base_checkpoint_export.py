import pytest
import torch

from scripts.base_checkpoint_export import load_source
from speck.checkpoint import save


def test_base_export_loads_checkpoint_with_provenance(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    metadata = {
        "step": 3,
        "training_phase": "base",
        "config": {},
        "resolved": {"steps": 3},
        "partial": False,
    }
    save(checkpoints, 3, {"weight": torch.tensor(3.0)}, {}, metadata)

    state, loaded, source, provenance = load_source(checkpoints, None)
    assert state["weight"].item() == 3.0
    assert loaded == metadata
    assert source == "step 3"
    assert provenance["type"] == "checkpoint"
    assert provenance["checkpoint"]["step"] == 3


@pytest.mark.parametrize(
    "metadata",
    (
        {"step": 2, "training_phase": "base", "resolved": {"steps": 3}, "partial": True},
        {"step": 2, "training_phase": "base", "resolved": {"steps": 3}, "partial": False},
        {"step": 2, "training_phase": "base", "resolved": {}, "partial": False},
    ),
)
def test_base_export_rejects_partial_or_nonfinal_checkpoint(tmp_path, metadata):
    checkpoints = tmp_path / "checkpoints"
    save(checkpoints, 2, {"weight": torch.tensor(2.0)}, {}, metadata)

    with pytest.raises(ValueError, match="non-partial resolved final"):
        load_source(checkpoints, 2)
