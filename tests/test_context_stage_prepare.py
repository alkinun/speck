import copy
import json
from pathlib import Path

import pytest

from scripts.context_stage_prepare import stage_configs
from speck.config import load_experiment

experiment = Path(__file__).parents[1] / "experiments" / "Speck1-140M"


def parent_inputs():
    configs = load_experiment(experiment, "model", "train")
    metadata = {"config": copy.deepcopy(configs["model"])}
    return configs, metadata


def test_context_stage_updates_only_positional_model_settings():
    configs, metadata = parent_inputs()
    model, train = stage_configs(
        configs,
        metadata,
        sequence_length=131_072,
        train_tokens=1_000_000_000,
        lr=1e-4,
        rope_theta=1_000_000.0,
        rope_scaling_factor=2.0,
        loss_backend="liger",
        activation_checkpointing=True,
        wandb_group="context-32k",
        run="SpeckLC-128K",
    )
    assert model["blocks"] == configs["model"]["blocks"]
    assert model["max_position_embeddings"] == 131_072
    assert model["rope_theta"] == 1_000_000.0
    assert model["rope_scaling_factor"] == 2.0
    assert train["sequence_length"] == 131_072
    assert train["training_phase"] == "context_extension"
    assert train["device_batch_size"] == 1
    assert train["loss_backend"] == "liger"
    assert train["activation_checkpointing"] is True
    assert train["wandb_group"] == "context-32k"


def test_context_stage_rejects_invalid_geometry():
    configs, metadata = parent_inputs()
    with pytest.raises(ValueError, match="sequence length"):
        stage_configs(
            configs,
            metadata,
            sequence_length=0,
            train_tokens=1,
            lr=1e-4,
            run="invalid",
        )
    with pytest.raises(ValueError, match="loss backend"):
        stage_configs(
            configs,
            metadata,
            sequence_length=32_768,
            train_tokens=1,
            lr=1e-4,
            loss_backend="unknown",
            run="invalid",
        )


def test_context_stage_output_is_json_serializable():
    configs, metadata = parent_inputs()
    values = stage_configs(
        configs,
        metadata,
        sequence_length=32_768,
        train_tokens=100_000_000,
        lr=1e-4,
        run="stage",
    )
    json.dumps(values)
