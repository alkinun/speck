import json
from dataclasses import replace
from pathlib import Path

import pytest

from speck.model.architecture import (
    ArchitectureConfig,
    AttentionSpec,
    BlockConfig,
    BlockGroup,
    KimiDeltaAttentionSpec,
    StageConfig,
    SwiGLUSpec,
)

REFERENCE_MODEL = Path(__file__).resolve().parents[2] / "experiments/qualification/model.json"


def test_main_architecture_round_trips():
    raw = json.loads(REFERENCE_MODEL.read_text())
    config = ArchitectureConfig.from_dict(raw)
    assert config.logical_depth == 24
    assert config.expected_parameters == 1_195_884_576
    assert config.embedding_size == 2048
    assert config.tie_word_embeddings is True
    assert config.export()["tie_word_embeddings"] is True
    assert ArchitectureConfig.from_dict(config.export()).settings() == config.settings()
    mixers = [invocation.block.stages[0].branches[0].kind for invocation in config.execution_plan]
    assert mixers.count("attention") == 6
    assert mixers.count("kimi_delta_attention") == 18
    assert mixers[:4] == ["kimi_delta_attention"] * 3 + ["attention"]


def test_legacy_default_attention_options_load_and_others_are_rejected():
    legacy = {
        "kind": "attention",
        "head_dim": 4,
        "num_key_value_heads": 1,
        "scope": "global",
        "window_size": None,
        "output_gate": "none",
        "memory": None,
        "memory_role": "none",
    }
    stage = {"branches": [legacy]}
    assert StageConfig.from_dict(stage).branches == (AttentionSpec(4, 1),)
    with pytest.raises(ValueError, match="unsupported attention option: scope"):
        StageConfig.from_dict({"branches": [{**legacy, "scope": "sliding", "window_size": 8}]})
    with pytest.raises(ValueError, match="unsupported attention option: output_gate"):
        StageConfig.from_dict({"branches": [{**legacy, "output_gate": "headwise"}]})


def test_missing_and_explicit_tied_configs_normalize_identically_and_untied_is_rejected():
    raw = json.loads(REFERENCE_MODEL.read_text())
    raw.pop("tie_word_embeddings")
    missing = ArchitectureConfig.from_dict(raw)
    explicit = ArchitectureConfig.from_dict({**raw, "tie_word_embeddings": True})

    assert missing == explicit
    assert missing.settings() == explicit.settings()
    with pytest.raises(ValueError, match="untied word embeddings are not supported"):
        ArchitectureConfig.from_dict({**raw, "tie_word_embeddings": False})
    with pytest.raises(ValueError, match="tie_word_embeddings must be boolean"):
        ArchitectureConfig.from_dict({**raw, "tie_word_embeddings": 1})


def test_unshared_repetitions_preserve_grouping_identity():
    block = BlockConfig(8, (StageConfig((SwiGLUSpec(16),)),))
    repeated = ArchitectureConfig((BlockGroup(block, repeat=2),), 8, vocab_size=16)
    expanded = ArchitectureConfig(
        (BlockGroup(block), BlockGroup(block)),
        8,
        vocab_size=16,
    )
    assert repeated.settings() != expanded.settings()
    assert ArchitectureConfig.from_dict(repeated.settings()) == repeated


def test_shared_blocks_keep_distinct_execution_state_identity():
    block = BlockConfig(
        8,
        (
            StageConfig((AttentionSpec(4, 1),)),
            StageConfig((SwiGLUSpec(16),)),
        ),
    )
    config = ArchitectureConfig(
        (BlockGroup(block, repeat=2, weight_sharing="all"),),
        8,
        vocab_size=16,
    )
    first, second = config.execution_plan
    assert first.weight_key == second.weight_key
    assert first.occurrence_index != second.occurrence_index


def test_focused_hybrid_grammar_round_trips():
    block = BlockConfig(
        16,
        (
            StageConfig((KimiDeltaAttentionSpec(8, 8, 1, 2, conv_kernel_size=3),)),
            StageConfig((AttentionSpec(8, 1, rope_dim=0),)),
            StageConfig((SwiGLUSpec(32),)),
        ),
    )
    config = ArchitectureConfig((BlockGroup(block),), 16, vocab_size=32)
    assert ArchitectureConfig.from_dict(config.export()) == config


def test_kimi_delta_attention_grammar_round_trips():
    operation = KimiDeltaAttentionSpec(
        8, 8, 2, 4, conv_kernel_size=3, output_gate_activation="silu"
    )
    block = BlockConfig(16, (StageConfig((operation,)),))
    config = ArchitectureConfig((BlockGroup(block),), 16, vocab_size=32)
    assert ArchitectureConfig.from_dict(config.export()) == config
    assert (
        config.export()["blocks"][0]["block"]["stages"][0]["branches"][0]["output_gate_activation"]
        == "silu"
    )


def test_kimi_delta_attention_default_and_explicit_sigmoid_serialize_as_legacy_config():
    default = KimiDeltaAttentionSpec(8, 8, 2, 4, conv_kernel_size=3)
    explicit = KimiDeltaAttentionSpec(
        8, 8, 2, 4, conv_kernel_size=3, output_gate_activation="sigmoid"
    )
    assert default == explicit

    config = ArchitectureConfig(
        (BlockGroup(BlockConfig(16, (StageConfig((explicit,)),))),),
        16,
        vocab_size=32,
    )
    branch = config.export()["blocks"][0]["block"]["stages"][0]["branches"][0]
    assert "output_gate_activation" not in branch
    assert ArchitectureConfig.from_dict(config.export()) == config


def test_kimi_delta_attention_rejects_unknown_output_gate_activation():
    with pytest.raises(ValueError, match="output gate activation"):
        KimiDeltaAttentionSpec(4, 4, 2, 4, output_gate_activation="relu")


def test_kimi_delta_attention_requires_supported_head_geometry():
    with pytest.raises(ValueError, match="value heads must be divisible"):
        KimiDeltaAttentionSpec(4, 4, 2, 3)
    with pytest.raises(ValueError, match="equal key and value"):
        KimiDeltaAttentionSpec(4, 8, 2, 4)


def test_attention_shape_invariants_are_strict():
    with pytest.raises(ValueError, match="divisible by attention"):
        BlockConfig(10, (StageConfig((AttentionSpec(4, 1),)),))
    with pytest.raises(ValueError, match="divisible by KV"):
        BlockConfig(12, (StageConfig((AttentionSpec(4, 2),)),))
    with pytest.raises(ValueError, match="RoPE dimensions"):
        AttentionSpec(8, 1, rope_dim=3)
    with pytest.raises(ValueError, match="RoPE dimensions"):
        AttentionSpec(8, 1, rope_dim=10)


def test_parallel_stage_kinds_must_be_unique():
    with pytest.raises(ValueError, match="must be unique"):
        StageConfig((SwiGLUSpec(16), SwiGLUSpec(32)))


def test_active_parameter_expectation_cannot_exceed_total():
    block = BlockConfig(8, (StageConfig((SwiGLUSpec(16),)),))
    with pytest.raises(ValueError, match="cannot exceed"):
        ArchitectureConfig(
            (BlockGroup(block),),
            8,
            vocab_size=16,
            expected_parameters=100,
            expected_active_parameters=101,
        )


@pytest.mark.parametrize(
    "field", ("rope_theta", "rope_scaling_factor", "rms_norm_eps", "initializer_range")
)
@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf"), True, "1", None))
def test_model_scaling_rejects_non_finite_or_non_numeric_values(field, value):
    block = BlockConfig(8, (StageConfig((SwiGLUSpec(16),)),))
    with pytest.raises(ValueError, match=field):
        ArchitectureConfig((BlockGroup(block),), 8, vocab_size=16, **{field: value})


@pytest.mark.parametrize(
    "config,field",
    (
        (AttentionSpec(4, 1), "head_dim"),
        (AttentionSpec(4, 1), "num_key_value_heads"),
        (AttentionSpec(4, 1), "rope_dim"),
        (KimiDeltaAttentionSpec(4, 4, 1, 1), "key_head_dim"),
        (KimiDeltaAttentionSpec(4, 4, 1, 1), "num_value_heads"),
        (KimiDeltaAttentionSpec(4, 4, 1, 1), "value_head_dim"),
        (KimiDeltaAttentionSpec(4, 4, 1, 1), "conv_kernel_size"),
        (SwiGLUSpec(16), "intermediate_size"),
        (BlockConfig(8, (StageConfig((SwiGLUSpec(16),)),)), "hidden_size"),
        (BlockGroup(BlockConfig(8, (StageConfig((SwiGLUSpec(16),)),))), "repeat"),
    ),
)
@pytest.mark.parametrize("value", (True, 4.0, "4"))
def test_architecture_dimensions_reject_non_integer_types(config, field, value):
    with pytest.raises(ValueError):
        replace(config, **{field: value})


@pytest.mark.parametrize(
    "field",
    (
        "embedding_size",
        "vocab_size",
        "max_position_embeddings",
        "bos_token_id",
        "eos_token_id",
        "expected_parameters",
        "expected_active_parameters",
    ),
)
def test_model_integer_fields_reject_boolean_values(field):
    config = ArchitectureConfig.from_dict(json.loads(REFERENCE_MODEL.read_text()))
    with pytest.raises(ValueError, match=field):
        replace(config, **{field: True})


def test_model_builder_preserves_explicit_reserved_vocabulary_rows():
    from speck.model import build_model

    settings = {
        "blocks": [
            {
                "block": {
                    "hidden_size": 8,
                    "stages": [{"branches": [{"kind": "swiglu", "intermediate_size": 16}]}],
                }
            }
        ],
        "embedding_size": 8,
        "vocab_size": 19,
    }
    model = build_model(settings, vocab_size=16)
    assert model.config.vocab_size == 19
    assert model.embed_tokens.weight.shape == (19, 8)
    with pytest.raises(ValueError, match="cover the tokenizer"):
        build_model({**settings, "vocab_size": 15}, vocab_size=16)
