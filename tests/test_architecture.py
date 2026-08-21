import json
from pathlib import Path

import pytest

from speck.architecture import (
    ArchitectureConfig,
    AttentionSpec,
    BlockConfig,
    BlockGroup,
    GatedCausalConvSpec,
    StageConfig,
    SwiGLUSpec,
)
from speck.model import Config


experiment = Path(__file__).parents[1] / "experiments" / "speck00-200m"


def test_v2_architecture_round_trips_through_v3():
    raw = json.loads((experiment / "model.json").read_text())
    v2 = Config.from_dict(raw)
    v3 = ArchitectureConfig.from_dict(raw)
    assert v3.logical_depth == 11
    assert v3.unique_parameter_blocks == 11
    assert v3.expected_parameters == 182_206_848
    assert v3.to_v2().settings() == v2.settings()
    assert ArchitectureConfig.from_dict(v3.export()) == v3
    assert v3.execution_plan[0].block.stages[0].branches[0].kind == "swiglu"
    assert v3.execution_plan[3].block.stages[0].branches[0].kind == "attention"


def test_unshared_repetitions_have_one_canonical_identity():
    block = BlockConfig(8, (StageConfig((SwiGLUSpec(16),)),))
    repeated = ArchitectureConfig((BlockGroup(block, repeat=2),), 8, vocab_size=16)
    expanded = ArchitectureConfig(
        (BlockGroup(block), BlockGroup(block)),
        8,
        vocab_size=16,
    )
    assert repeated.digest == expanded.digest
    assert repeated.settings() == expanded.settings()


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
    assert config.unique_parameter_blocks == 1


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
    with pytest.raises(ValueError, match="cannot be represented"):
        config.to_v2()


def test_attention_shape_invariants_are_strict():
    with pytest.raises(ValueError, match="divisible by attention"):
        BlockConfig(10, (StageConfig((AttentionSpec(4, 1),)),))
    with pytest.raises(ValueError, match="divisible by kv"):
        BlockConfig(12, (StageConfig((AttentionSpec(4, 2),)),))


def test_parallel_stage_kinds_must_be_unique():
    with pytest.raises(ValueError, match="must be unique"):
        StageConfig((SwiGLUSpec(16), SwiGLUSpec(32)))
