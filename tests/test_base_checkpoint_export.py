import pytest
import torch

from scripts.base_checkpoint_export import load_source
from scripts.checkpoint_average import write_average
from speck.checkpoint import save


def test_base_export_loads_checkpoint_or_average(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    metadata = {
        "step": 3,
        "training_phase": "base",
        "config": {},
        "resolved": {},
    }
    save(checkpoints, 3, {"weight": torch.tensor(3.0)}, {}, metadata)

    state, loaded, source = load_source(checkpoints, None)
    assert state["weight"].item() == 3.0
    assert loaded == metadata
    assert source == "step 3"

    average = tmp_path / "average"
    average_metadata = {
        "format": "speck_model_average",
        "format_version": 1,
        "config": {},
        "resolved": {},
    }
    write_average(average, {"weight": torch.tensor(4.0)}, average_metadata)

    state, loaded, source = load_source(average, None)
    assert state["weight"].item() == 4.0
    assert loaded == average_metadata
    assert source == "average"
    with pytest.raises(ValueError, match="--step"):
        load_source(average, 3)
