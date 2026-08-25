import pytest
import torch

from scripts.model_publish import release_config, release_state


def metadata():
    return {
        "config": {
            "blocks": [
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
                }
            ],
            "embedding_size": 3,
            "vocab_size": 7,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "max_position_embeddings": 32,
            "rope_theta": 10_000.0,
            "rms_norm_eps": 1e-5,
            "initializer_range": 0.02,
        },
        "resolved": {"parameters": 100},
    }


def test_release_config_adds_transformers_metadata():
    result = release_config(metadata())

    assert result["architectures"] == ["SpeckForCausalLM"]
    assert result["auto_map"]["AutoModelForCausalLM"].endswith("SpeckForCausalLM")
    assert result["dtype"] == "bfloat16"
    assert result["num_attention_heads"] == 2
    assert result["num_hidden_layers"] == 1
    assert result["expected_parameters"] == 100


def test_release_state_converts_to_bf16_and_omits_tied_head():
    embedding = torch.randn(7, 3)
    result = release_state(
        {
            "embed_tokens.weight": embedding,
            "lm_head.weight": embedding,
            "norm.weight": torch.ones(3),
        }
    )

    assert set(result) == {"embed_tokens.weight", "norm.weight"}
    assert all(tensor.dtype == torch.bfloat16 for tensor in result.values())


def test_release_state_rejects_untied_head():
    with pytest.raises(ValueError, match="not tied"):
        release_state(
            {
                "embed_tokens.weight": torch.zeros(2, 2),
                "lm_head.weight": torch.ones(2, 2),
            }
        )
