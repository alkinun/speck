import copy
import json
from pathlib import Path

import pytest

from scripts.context_stage_prepare import promote_global_attention_layers, stage_configs
from speck.architecture import ArchitectureConfig, AttentionSpec
from speck.config import load_experiment

experiment = Path(__file__).parents[1] / "experiments" / "Speck1-140M"
local_experiment = (
    Path(__file__).parents[1]
    / "experiments"
    / "SpeckLC-150M-MixerScreen-131M"
    / "gdn-local"
)


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


def test_context_stage_can_promote_selected_sliding_attention_layers():
    model = load_experiment(local_experiment, "model")["model"]
    promoted = ArchitectureConfig.from_dict(promote_global_attention_layers(model, (11, 19)))
    scopes = {
        invocation.occurrence_index: branch.scope
        for invocation in promoted.execution_plan
        for stage in invocation.block.stages
        for branch in stage.branches
        if isinstance(branch, AttentionSpec)
    }
    assert scopes == {3: "sliding", 7: "sliding", 11: "global", 15: "sliding", 19: "global"}

    with pytest.raises(ValueError, match="not sliding-attention"):
        promote_global_attention_layers(model, (10,))
