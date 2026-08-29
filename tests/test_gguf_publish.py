import pytest
import torch

from scripts.gguf_publish import (
    transform_state,
    transformed_config,
    transformed_parameter_count,
    validate_config,
)


def config():
    return {
        "architectures": ["SpeckForCausalLM"],
        "blocks": [
            {
                "block": {
                    "hidden_size": 4,
                    "stages": [
                        {
                            "branches": [
                                {
                                    "kind": "gated_causal_conv",
                                    "inner_size": 2,
                                    "kernel_size": 3,
                                }
                            ]
                        },
                        {"branches": [{"kind": "swiglu", "intermediate_size": 8}]},
                    ],
                }
            },
            {
                "block": {
                    "hidden_size": 4,
                    "stages": [
                        {
                            "branches": [
                                {
                                    "kind": "attention",
                                    "head_dim": 2,
                                    "num_key_value_heads": 1,
                                }
                            ]
                        },
                        {"branches": [{"kind": "swiglu", "intermediate_size": 8}]},
                    ],
                }
            },
        ],
        "bos_token_id": 1,
        "eos_token_id": 2,
        "max_position_embeddings": 32,
        "embedding_size": 3,
        "expected_parameters": 347,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10_000.0,
        "vocab_size": 7,
    }


def state():
    torch.manual_seed(1)
    values = {
        "embed_tokens.weight": torch.randn(7, 3),
        "adapters.0.weight": torch.randn(4, 3),
        "output_projection.weight": torch.randn(3, 4),
        "norm.weight": torch.randn(4),
    }
    for index, kind in enumerate(("conv", "attention")):
        prefix = f"cores.group_{index}_repeat_0"
        operator = f"{prefix}.stages.0.branches.0"
        feed_forward = f"{prefix}.stages.1.branches.0"
        values[f"{operator}.norm.weight"] = torch.randn(4)
        values[f"{feed_forward}.norm.weight"] = torch.randn(4)
        values[f"{feed_forward}.operation.gate_proj.weight"] = torch.randn(8, 4)
        values[f"{feed_forward}.operation.up_proj.weight"] = torch.randn(8, 4)
        values[f"{feed_forward}.operation.down_proj.weight"] = torch.randn(4, 8)
        if kind == "conv":
            values[f"{operator}.operation.input_projection.weight"] = torch.randn(6, 4)
            values[f"{operator}.operation.output_projection.weight"] = torch.randn(4, 2)
            values[f"{operator}.operation.kernel"] = torch.randn(2, 1, 3)
        else:
            values[f"{operator}.operation.q_proj.weight"] = torch.randn(4, 4)
            values[f"{operator}.operation.k_proj.weight"] = torch.randn(2, 4)
            values[f"{operator}.operation.v_proj.weight"] = torch.randn(2, 4)
            values[f"{operator}.operation.o_proj.weight"] = torch.randn(4, 4)
            values[f"{operator}.operation.q_norm.weight"] = torch.randn(2)
            values[f"{operator}.operation.k_norm.weight"] = torch.randn(2)
    return values


def test_transform_folds_adapters_and_pads_convolution():
    source = state()
    transformed, layout = transform_state(source, config())

    assert torch.allclose(
        transformed["model.embed_tokens.weight"],
        source["embed_tokens.weight"] @ source["adapters.0.weight"].T,
    )
    assert torch.allclose(
        transformed["lm_head.weight"],
        source["embed_tokens.weight"] @ source["output_projection.weight"],
    )
    padded_input = transformed["model.layers.0.conv.in_proj.weight"]
    assert padded_input.shape == (12, 4)
    assert torch.equal(
        padded_input[0:2],
        source["cores.group_0_repeat_0.stages.0.branches.0.operation.input_projection.weight"][0:2],
    )
    assert torch.count_nonzero(padded_input[2:4]) == 0
    assert transformed["model.layers.0.conv.conv.weight"].shape == (4, 1, 3)
    assert layout["layer_types"] == ["conv", "full_attention"]


def test_transformed_config_selects_attention_layers_with_kv_heads():
    source = config()
    layout = validate_config(source)

    result = transformed_config(source, layout)

    assert result["architectures"] == ["Lfm2ForCausalLM"]
    assert result["layer_types"] == ["conv", "full_attention"]
    assert result["num_attention_heads"] == 2
    assert result["num_key_value_heads"] == 1
    assert result["conv_L_cache"] == 3


def test_transformed_parameter_count_includes_padded_convolution():
    source = config()
    layout = validate_config(source)

    assert transformed_parameter_count(source, layout) == sum(
        tensor.numel() for tensor in transform_state(state(), source)[0].values()
    )


def test_transform_rejects_unmapped_checkpoint_tensors():
    source = state()
    source["unexpected.weight"] = torch.ones(1)

    with pytest.raises(ValueError, match="unmapped tensors"):
        transform_state(source, config())
