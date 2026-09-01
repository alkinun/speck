import json
from pathlib import Path

import pytest

from speck.architecture import (
    ArchitectureConfig,
    AttentionSpec,
    BlockConfig,
    BlockGroup,
    GatedCausalConvSpec,
    GatedDeltaNetSpec,
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
    operation = GatedDeltaNetSpec(4, 8, 2, 4, conv_kernel_size=3)
    block = BlockConfig(16, (StageConfig((operation,)),))
    config = ArchitectureConfig((BlockGroup(block),), 16, vocab_size=32)
    assert ArchitectureConfig.from_dict(config.export()) == config


def test_gated_deltanet_requires_grouped_value_heads():
    with pytest.raises(ValueError, match="value heads must be divisible"):
        GatedDeltaNetSpec(4, 4, 2, 3)


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
