import pytest
import torch

from scripts.base_checkpoint_export import (
    has_routed_layers,
    load_source,
    patch_moe_configuration_source,
    patch_moe_modeling_source,
)
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


def test_base_export_detects_routed_architecture():
    metadata = {
        "config": {
            "blocks": [
                {
                    "block": {
                        "hidden_size": 4,
                        "stages": [
                            {
                                "branches": [
                                    {
                                        "kind": "routed_swiglu",
                                        "intermediate_size": 8,
                                        "num_experts": 4,
                                        "top_k": 2,
                                    }
                                ]
                            }
                        ],
                    }
                }
            ],
            "embedding_size": 4,
            "vocab_size": 8,
        }
    }

    assert has_routed_layers(metadata)


def test_moe_transformers_patches_reject_pinned_source_drift():
    with pytest.raises(ValueError, match="unexpected SwiGLU specification"):
        patch_moe_modeling_source("changed")
    with pytest.raises(ValueError, match="unexpected operation validation"):
        patch_moe_configuration_source("changed")
