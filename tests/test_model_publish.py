import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from scripts.model_publish import (
    CODE_FILES,
    MODEL_FORWARD_SETUP,
    MODEL_GENERATION_PREPARE,
    MODEL_IMPORT,
    MODEL_POSITION_CHECK,
    PADDING_DESTINATION,
    patch_generation_source,
    patch_modeling_source,
    prepare_current_release_code,
    prepare_release_code,
    release_config,
    release_state,
)
from speck.architecture import ArchitectureConfig
from speck.model import SpeckForCausalLM


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

    assert set(result) == {"native.embed_tokens.weight", "native.norm.weight"}
    assert all(tensor.dtype == torch.bfloat16 for tensor in result.values())


def test_release_state_rejects_untied_head():
    with pytest.raises(ValueError, match="not tied"):
        release_state(
            {
                "embed_tokens.weight": torch.zeros(2, 2),
                "lm_head.weight": torch.ones(2, 2),
            }
        )


def model_source():
    return (
        MODEL_IMPORT
        + "\nclass Model:\n"
        + "    def forward(self):\n"
        + MODEL_FORWARD_SETUP
        + "        position = 0\n"
        + "        expected_positions = torch.arange(position, position + length)\n"
        + MODEL_POSITION_CHECK
    )


def test_modeling_patch_adds_right_padding_support():
    result = patch_modeling_source(model_source())

    assert "from .padding_speck import validate_right_padding" in result
    assert "has_padding = validate_right_padding" in result
    assert "right-padded inputs require use_cache=False" in result
    assert "position_ids[valid]" in result
    assert "Speck does not support padded inputs" not in result


def test_modeling_patch_rejects_source_drift():
    with pytest.raises(ValueError, match="unexpected configuration import"):
        patch_modeling_source("changed source")


def test_generation_patch_slices_cached_attention_mask():
    source = "class Model:\n    def prepare(self):\n" + MODEL_GENERATION_PREPARE
    result = patch_generation_source(source)

    assert 'model_inputs["attention_mask"] = current_mask[:, -current.size(1) :]' in result
    assert "return model_inputs" in result


def test_generation_patch_rejects_source_drift():
    with pytest.raises(ValueError, match="unexpected generation preparation"):
        patch_generation_source("changed source")


def test_prepare_release_code_copies_patched_support(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    for filename in CODE_FILES:
        value = (
            model_source() + MODEL_GENERATION_PREPARE
            if filename == "modeling_speck.py"
            else filename
        )
        (source / filename).write_text(value, encoding="utf-8")

    prepare_release_code(source, output)

    assert "validate_right_padding" in (output / "modeling_speck.py").read_text()
    assert "current_mask[:, -current.size(1) :]" in (output / "modeling_speck.py").read_text()
    assert (output / PADDING_DESTINATION).read_text() == Path(
        "speck/transformers_padding.py"
    ).read_text()


def test_current_release_code_vendors_the_native_architecture(tmp_path):
    prepare_current_release_code(tmp_path)
    native = (tmp_path / "native_speck.py").read_text()
    assert "from .architecture_speck import (" in native
    assert "class GatedDeltaNet" in native
    assert (tmp_path / "architecture_speck.py").is_file()
    assert (tmp_path / "configuration_speck.py").is_file()
    assert (tmp_path / "modeling_speck.py").is_file()


def assert_current_transformers_parity(tmp_path, values):
    transformers = pytest.importorskip("transformers")
    architecture = ArchitectureConfig.from_dict(values["config"])
    native = SpeckForCausalLM(architecture)
    native.init_weights()
    native.to(torch.bfloat16)
    native.eval()
    values["resolved"]["parameters"] = native.parameter_count()
    prepare_current_release_code(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(release_config(values)), encoding="utf-8")
    save_file(release_state(native.state_dict()), tmp_path / "model.safetensors")
    exported = transformers.AutoModelForCausalLM.from_pretrained(
        tmp_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    tokens = torch.randint(0, architecture.vocab_size, (1, 8))
    with torch.no_grad():
        expected = native(tokens)
        actual = exported(input_ids=tokens, use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_current_transformers_wrapper_matches_native_logits(tmp_path):
    assert_current_transformers_parity(tmp_path, metadata())


def test_current_transformers_wrapper_exports_gated_deltanet(tmp_path):
    values = metadata()
    values["config"]["blocks"] = [
        {
            "block": {
                "hidden_size": 4,
                "stages": [
                    {
                        "branches": [
                            {
                                "kind": "gated_deltanet",
                                "key_head_dim": 2,
                                "value_head_dim": 2,
                                "num_key_heads": 1,
                                "num_value_heads": 2,
                                "conv_kernel_size": 3,
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
                                "rope_dim": 0,
                            }
                        ]
                    },
                    {"branches": [{"kind": "swiglu", "intermediate_size": 8}]},
                ],
            }
        },
    ]
    assert_current_transformers_parity(tmp_path, values)
