import json
from pathlib import Path

import torch

from speck.architecture import (
    ArchitectureConfig,
    AttentionSpec,
    BlockConfig,
    BlockGroup,
    GatedCausalConvSpec,
    StageConfig,
    SwiGLUSpec,
)
from speck.model_v3 import SpeckV3ForCausalLM
from speck.search.architecture_v3 import (
    V3SearchSpace,
    architecture_distance,
    available_mutations,
    crossover,
    mutate,
    parameter_count,
    quantized_weight_bytes,
    sample_architecture,
    state_bytes,
)


experiment = Path(__file__).parents[1] / "experiments" / "speck00-200m"


def space():
    return V3SearchSpace(
        min_logical_depth=2,
        max_logical_depth=6,
        hidden_sizes=(8, 12, 16),
        intermediate_sizes=(16, 24, 32),
        head_dims=(4, 8),
        kv_heads=(1, 2),
        sliding_windows=(2, 4),
        conv_kernel_sizes=(2, 3),
        conv_inner_sizes=(8, 16),
        repeat_counts=(1, 2),
    )


def base():
    block = BlockConfig(
        8,
        (
            StageConfig((AttentionSpec(4, 1, "sliding", 2),)),
            StageConfig((SwiGLUSpec(16),)),
        ),
    )
    convolution = BlockConfig(
        8,
        (
            StageConfig((GatedCausalConvSpec(8, 2),)),
            StageConfig((SwiGLUSpec(16),)),
        ),
    )
    return ArchitectureConfig(
        (BlockGroup(block), BlockGroup(convolution)),
        8,
        vocab_size=16,
        max_position_embeddings=16,
    )


def test_v3_static_parameter_count_matches_the_runtime():
    config = ArchitectureConfig.from_dict(
        json.loads((experiment / "model.json").read_text())
    )
    with torch.device("meta"):
        model = SpeckV3ForCausalLM(config)
    assert parameter_count(config) == model.parameter_count() == 182_206_848


def test_shared_weights_reduce_parameters_but_not_state():
    block = base().blocks[0].block
    shared = ArchitectureConfig(
        (BlockGroup(block, repeat=2, weight_sharing="all"),),
        8,
        vocab_size=16,
    )
    unshared = ArchitectureConfig(
        (BlockGroup(block, repeat=2, weight_sharing="none"),),
        8,
        vocab_size=16,
    )
    assert parameter_count(shared) < parameter_count(unshared)
    assert state_bytes(shared, 8) == state_bytes(unshared, 8)
    assert quantized_weight_bytes(shared, group_size=4) < quantized_weight_bytes(
        unshared, group_size=4
    )


def test_state_bytes_include_bounded_attention_and_convolution():
    config = base()
    expected_attention = 2 * 2 * 1 * 4 * 2
    expected_convolution = 8 * 1 * 2
    assert state_bytes(config, context=8) == expected_attention + expected_convolution


def test_sampling_and_mutation_produce_valid_distinct_architectures():
    first = sample_architecture(base(), space(), 1)
    second = sample_architecture(base(), space(), 2)
    assert first.digest != second.digest
    assert 2 <= first.logical_depth <= 6
    result = mutate(base(), space(), 3, "change_mixer")
    assert result.config.digest != base().digest
    assert result.operation["operator"] == "change_mixer"
    assert parameter_count(result.config) == SpeckV3ForCausalLM(result.config).parameter_count()


def test_crossover_and_distance_cover_hybrid_layouts():
    left = sample_architecture(base(), space(), 4)
    right = sample_architecture(base(), space(), 5)
    child = crossover(left, right, space(), 6).config
    assert 2 <= child.logical_depth <= 6
    assert architecture_distance(left, left, space()) == 0
    assert architecture_distance(left, right, space()) > 0


def test_every_available_mutation_changes_the_architecture():
    original = base()
    original = ArchitectureConfig(
        (
            BlockGroup(
                original.blocks[0].block,
                repeat=2,
                weight_sharing="all",
            ),
            original.blocks[1],
        ),
        original.embedding_size,
        vocab_size=original.vocab_size,
        max_position_embeddings=original.max_position_embeddings,
    )
    for index, operator in enumerate(available_mutations(original, space())):
        result = mutate(original, space(), 100 + index, operator)
        assert result.config.digest != original.digest
        assert parameter_count(result.config) == SpeckV3ForCausalLM(
            result.config
        ).parameter_count()
