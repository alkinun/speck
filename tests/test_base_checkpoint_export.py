import torch

from scripts.base_checkpoint_export import load_source
from speck.checkpoint import save


def test_base_export_loads_checkpoint_with_provenance(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    metadata = {
        "step": 3,
        "training_phase": "base",
        "config": {},
        "resolved": {},
    }
    save(checkpoints, 3, {"weight": torch.tensor(3.0)}, {}, metadata)

    state, loaded, source, provenance = load_source(checkpoints, None)
    assert state["weight"].item() == 3.0
    assert loaded == metadata
    assert source == "step 3"
    assert provenance["type"] == "checkpoint"
    assert provenance["checkpoint"]["step"] == 3
