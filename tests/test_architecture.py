import json
from dataclasses import replace
from pathlib import Path

import pytest

from speck.architecture import (
    ArchitectureConfig,
    AttentionSpec,
    BlockConfig,
    BlockGroup,
    GatedCausalConvSpec,
    GatedDeltaNetSpec,
    KimiDeltaAttentionSpec,
    RoutedSwiGLUSpec,
    StageConfig,
    SwiGLUSpec,
)

experiment = Path(__file__).parents[1] / "experiments" / "Speck1-140M"


def test_main_architecture_round_trips():
    raw = json.loads((experiment / "model.json").read_text())
    config = ArchitectureConfig.from_dict(raw)
    assert config.logical_depth == 18
    assert config.expected_parameters == 140_652_288
    assert config.embedding_size == 640
    assert ArchitectureConfig.from_dict(config.export()).settings() == config.settings()
    mixers = [invocation.block.stages[0].branches[0].kind for invocation in config.execution_plan]
    assert mixers.count("attention") == 8
    assert mixers.count("gated_causal_conv") == 10
    assert config.execution_plan[0].block.stages[0].branches[0].kind == "gated_causal_conv"
    assert config.execution_plan[1].block.stages[0].branches[0].kind == "attention"


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
            StageConfig((GatedCausalConvSpec(16, 3),)),
            StageConfig((AttentionSpec(8, 1, "sliding", 32),)),
            StageConfig((SwiGLUSpec(32),)),
        ),
    )
    config = ArchitectureConfig((BlockGroup(block),), 16, vocab_size=32)
    assert ArchitectureConfig.from_dict(config.export()) == config


def test_gated_deltanet_grammar_round_trips():
    operation = GatedDeltaNetSpec(
        4,
        8,
        2,
        4,
        conv_kernel_size=3,
        output_gate_activation="sigmoid",
        decay_initialization="fla",
    )
    block = BlockConfig(16, (StageConfig((operation,)),))
    config = ArchitectureConfig((BlockGroup(block),), 16, vocab_size=32)
    assert ArchitectureConfig.from_dict(config.export()) == config


def test_gated_deltanet_requires_grouped_value_heads():
    with pytest.raises(ValueError, match="value heads must be divisible"):
        GatedDeltaNetSpec(4, 4, 2, 3)


def test_gated_deltanet_rejects_unknown_output_gate_activation():
    with pytest.raises(ValueError, match="output gate activation"):
        GatedDeltaNetSpec(4, 4, 2, 4, output_gate_activation="relu")


def test_gated_deltanet_rejects_unknown_decay_initialization():
    with pytest.raises(ValueError, match="decay initialization"):
        GatedDeltaNetSpec(4, 4, 2, 4, decay_initialization="unknown")


def test_kimi_delta_attention_grammar_round_trips():
    operation = KimiDeltaAttentionSpec(8, 8, 2, 4, conv_kernel_size=3)
    block = BlockConfig(16, (StageConfig((operation,)),))
    config = ArchitectureConfig((BlockGroup(block),), 16, vocab_size=32)
    assert ArchitectureConfig.from_dict(config.export()) == config


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
    with pytest.raises(ValueError, match="output gate"):
        AttentionSpec(8, 1, output_gate="unknown")


def test_parallel_stage_kinds_must_be_unique():
    with pytest.raises(ValueError, match="must be unique"):
        StageConfig((SwiGLUSpec(16), SwiGLUSpec(32)))


def memory_layers(**overrides):
    writer = AttentionSpec(4, 1, memory="global", memory_role="write", **overrides)
    reader = AttentionSpec(4, 1, memory="global", memory_role="read", **overrides)
    return writer, reader


def memory_architecture(*layers):
    groups = tuple(
        BlockGroup(BlockConfig(8, (StageConfig((layer,)), StageConfig((SwiGLUSpec(16),)))))
        for layer in layers
    )
    return ArchitectureConfig(groups, 8, vocab_size=16)


def test_reader_attention_grammar_round_trips():
    writer, reader = memory_layers(rope_dim=0)
    config = memory_architecture(writer, reader, reader)
    assert ArchitectureConfig.from_dict(config.export()) == config
    assert writer.writes_memory and not writer.reads_memory
    assert reader.reads_memory and not reader.writes_memory


def test_attention_memory_label_and_role_are_declared_together():
    with pytest.raises(ValueError, match="declared together"):
        AttentionSpec(4, 1, memory="global")
    with pytest.raises(ValueError, match="declared together"):
        AttentionSpec(4, 1, memory_role="read")
    with pytest.raises(ValueError, match="memory role"):
        AttentionSpec(4, 1, memory="global", memory_role="borrow")


def test_attention_memory_requires_global_scope():
    with pytest.raises(ValueError, match="requires global scope"):
        AttentionSpec(4, 1, "sliding", 8, memory="global", memory_role="write")


def test_attention_memory_readers_must_follow_their_writer():
    writer, reader = memory_layers()
    with pytest.raises(ValueError, match="must follow their memory writer"):
        memory_architecture(reader, writer)
    with pytest.raises(ValueError, match="must follow their memory writer"):
        memory_architecture(reader)
    with pytest.raises(ValueError, match="must follow their memory writer"):
        memory_architecture(writer, AttentionSpec(4, 1, memory="other", memory_role="read"))


def test_each_attention_memory_keeps_exactly_one_writer():
    writer, _ = memory_layers()
    with pytest.raises(ValueError, match="exactly one writer"):
        memory_architecture(writer, writer)
    block = BlockConfig(8, (StageConfig((writer,)), StageConfig((SwiGLUSpec(16),))))
    with pytest.raises(ValueError, match="exactly one writer"):
        ArchitectureConfig((BlockGroup(block, repeat=2),), 8, vocab_size=16)


def test_attention_memory_readers_must_match_writer_key_geometry():
    writer = AttentionSpec(4, 1, memory="global", memory_role="write")
    with pytest.raises(ValueError, match="writer key geometry"):
        memory_architecture(writer, AttentionSpec(4, 2, memory="global", memory_role="read"))
    with pytest.raises(ValueError, match="writer key geometry"):
        memory_architecture(
            writer, AttentionSpec(4, 1, rope_dim=0, memory="global", memory_role="read")
        )


def test_attention_memory_readers_accept_equivalent_rope_declarations():
    writer = AttentionSpec(4, 1, rope_dim=4, memory="global", memory_role="write")
    reader = AttentionSpec(4, 1, memory="global", memory_role="read")
    assert memory_architecture(writer, reader).logical_depth == 2


@pytest.mark.parametrize(
    ("intermediate_size", "num_experts", "top_k"),
    ((0, 4, 1), (16, 0, 1), (16, 4, 0), (16, 4, 5)),
)
def test_routed_swiglu_shape_invariants_are_strict(intermediate_size, num_experts, top_k):
    with pytest.raises(ValueError, match="routed SwiGLU"):
        RoutedSwiGLUSpec(intermediate_size, num_experts, top_k)


def test_routed_swiglu_round_trips_and_accounts_only_selected_experts():
    block = BlockConfig(
        8,
        (StageConfig((RoutedSwiGLUSpec(16, num_experts=4, top_k=2),)),),
    )
    config = ArchitectureConfig(
        (BlockGroup(block, repeat=2, weight_sharing="all"),),
        embedding_size=8,
        vocab_size=16,
        expected_parameters=10_000,
        expected_active_parameters=8_464,
    )

    assert ArchitectureConfig.from_dict(config.export()) == config
    assert config.settings()["blocks"][0]["block"]["stages"][0]["branches"][0] == {
        "intermediate_size": 16,
        "kind": "routed_swiglu",
        "num_experts": 4,
        "top_k": 2,
    }
    assert config.active_parameter_count(10_000) == 10_000 - 2 * 3 * 8 * 16


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
        (AttentionSpec(4, 1, "sliding", 4), "window_size"),
        (GatedCausalConvSpec(8, 3), "inner_size"),
        (GatedCausalConvSpec(8, 3), "kernel_size"),
        (GatedDeltaNetSpec(4, 4, 1, 1), "key_head_dim"),
        (GatedDeltaNetSpec(4, 4, 1, 1), "num_value_heads"),
        (KimiDeltaAttentionSpec(4, 4, 1, 1), "value_head_dim"),
        (KimiDeltaAttentionSpec(4, 4, 1, 1), "conv_kernel_size"),
        (SwiGLUSpec(16), "intermediate_size"),
        (RoutedSwiGLUSpec(16, 4, 2), "num_experts"),
        (RoutedSwiGLUSpec(16, 4, 2), "top_k"),
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
    config = ArchitectureConfig.from_dict(json.loads((experiment / "model.json").read_text()))
    with pytest.raises(ValueError, match=field):
        replace(config, **{field: True})
