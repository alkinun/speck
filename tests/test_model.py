import json
from pathlib import Path

import pytest
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
from speck.model import (
    CombinedOptimizer,
    Linear,
    SpeckForCausalLM,
    build_model,
    initialize_backbone,
)

experiment = Path(__file__).parents[1] / "experiments" / "Speck1-140M"


def model_with(*stages, repeat=1, sharing="none"):
    block = BlockConfig(8, tuple(StageConfig((stage,)) for stage in stages))
    config = ArchitectureConfig(
        (BlockGroup(block, repeat=repeat, weight_sharing=sharing),),
        8,
        vocab_size=16,
        max_position_embeddings=16,
    )
    model = SpeckForCausalLM(config)
    model.init_weights()
    model.eval()
    return model


def cached_logits(model, tokens):
    state = model.state(length=tokens.size(1))
    values = [model(tokens[:, :1], state=state)]
    for index in range(1, tokens.size(1)):
        values.append(model(tokens[:, index : index + 1], state=state))
    return torch.cat(values, dim=1)


def test_linear_applies_configured_bias():
    layer = Linear(2, 1, bias=True)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[2.0, 3.0]]))
        layer.bias.fill_(5.0)

    assert torch.equal(layer(torch.tensor([[7.0, 11.0]])), torch.tensor([[52.0]]))


def test_main_model_parameter_count():
    settings = json.loads((experiment / "model.json").read_text())
    config = ArchitectureConfig.from_dict(settings)
    model = SpeckForCausalLM(config)
    assert model.parameter_count() == 140_652_288


def test_global_attention_cache_matches_full_forward():
    torch.manual_seed(1)
    model = model_with(AttentionSpec(4, 1), SwiGLUSpec(16))
    tokens = torch.randint(0, 16, (1, 8))
    assert torch.allclose(model(tokens), cached_logits(model, tokens), atol=1e-5)


def test_sliding_attention_cache_matches_full_forward():
    torch.manual_seed(2)
    model = model_with(AttentionSpec(4, 1, "sliding", 3), SwiGLUSpec(16))
    tokens = torch.randint(0, 16, (1, 8))
    state = model.state(length=8)
    assert state.allocated_bytes() == 1 * 1 * 3 * 4 * 4 * 2
    assert torch.allclose(model(tokens), cached_logits(model, tokens), atol=1e-5)


def test_convolution_state_matches_full_forward():
    torch.manual_seed(3)
    model = model_with(GatedCausalConvSpec(8, 3), SwiGLUSpec(16))
    tokens = torch.randint(0, 16, (1, 8))
    assert torch.allclose(model(tokens), cached_logits(model, tokens), atol=1e-5)


def test_muon_optimizer_routes_convolution_parameters_to_adamw():
    model = model_with(GatedCausalConvSpec(8, 3), SwiGLUSpec(16))
    optimizer = model.optimizer(name="muon")
    assert isinstance(optimizer, CombinedOptimizer)
    muon_parameters = {
        id(parameter)
        for group in optimizer.optimizers["muon"].param_groups
        for parameter in group["params"]
    }
    adamw_parameters = {
        id(parameter)
        for group in optimizer.optimizers["adamw"].param_groups
        for parameter in group["params"]
    }
    convolution_parameters = {
        id(parameter) for parameter in model.parameters() if parameter.ndim == 3
    }

    assert muon_parameters.isdisjoint(adamw_parameters)
    assert muon_parameters | adamw_parameters == {id(parameter) for parameter in model.parameters()}
    assert convolution_parameters <= adamw_parameters
    assert optimizer.optimizers["adamw"].param_groups[0]["weight_decay"] == 0.1

    tokens = torch.randint(0, 16, (2, 8))
    model(tokens, tokens).backward()
    optimizer.step()
    state = optimizer.state_dict()
    optimizer.load_state_dict(state)


def test_shared_blocks_keep_occurrence_state_separate():
    torch.manual_seed(4)
    model = model_with(
        AttentionSpec(4, 1, "sliding", 3),
        GatedCausalConvSpec(8, 3),
        repeat=2,
        sharing="all",
    )
    state = model.state(length=8)
    assert len(state.entries) == 4
    assert len(model.cores) == 1
    tokens = torch.randint(0, 16, (1, 8))
    assert torch.allclose(model(tokens), cached_logits(model, tokens), atol=1e-5)


def test_state_reset_replays_the_same_sequence():
    model = model_with(AttentionSpec(4, 1), GatedCausalConvSpec(8, 3))
    state = model.state(length=8)
    tokens = torch.randint(0, 16, (1, 4))
    first = model(tokens, state=state)
    state.reset()
    second = model(tokens, state=state)
    assert torch.equal(first, second)


@pytest.mark.parametrize("length", (0, -1, 17, True, 1.5))
def test_state_rejects_invalid_lengths(length):
    model = model_with(AttentionSpec(4, 1))
    with pytest.raises(ValueError, match="state length"):
        model.state(length=length)


@pytest.mark.parametrize("batch_size", (0, -1, True, 1.5))
def test_state_rejects_invalid_batch_sizes(batch_size):
    model = model_with(AttentionSpec(4, 1))
    with pytest.raises(ValueError, match="batch size"):
        model.state(batch_size=batch_size)


def test_build_model_uses_block_config():
    model = model_with(SwiGLUSpec(16), repeat=2)
    rebuilt = build_model(model.config.export(), 16)
    assert isinstance(rebuilt, SpeckForCausalLM)
    assert rebuilt.parameter_count() == model.parameter_count()
    rebuilt.load_state_dict(model.state_dict())


def test_resize_token_embeddings_preserves_and_reties_rows():
    model = model_with(SwiGLUSpec(16))
    original = model.embed_tokens.weight.detach().clone()
    parameters = model.parameter_count()

    model.resize_token_embeddings(19)

    assert model.config.vocab_size == 19
    assert model.config.expected_parameters is None
    assert model.embed_tokens.weight.shape == (19, 8)
    assert torch.equal(model.embed_tokens.weight[:16], original)
    assert model.lm_head.weight is model.embed_tokens.weight
    assert model.parameter_count() == parameters + 3 * 8
    with pytest.raises(ValueError, match="cannot shrink"):
        model.resize_token_embeddings(15)


def test_backbone_initialization_resets_the_complete_token_interface():
    torch.manual_seed(10)
    source_config = ArchitectureConfig(
        (
            BlockGroup(BlockConfig(12, (StageConfig((AttentionSpec(4, 1),)),))),
            BlockGroup(BlockConfig(8, (StageConfig((SwiGLUSpec(16),)),))),
        ),
        embedding_size=6,
        vocab_size=16,
        max_position_embeddings=8,
    )
    source = SpeckForCausalLM(source_config)
    source.init_weights()
    with torch.no_grad():
        for index, parameter in enumerate(source.parameters(), start=1):
            parameter.fill_(index)

    torch.manual_seed(20)
    target = SpeckForCausalLM(source_config)
    target.init_weights()
    fresh = {key: value.clone() for key, value in target.state_dict().items()}
    report = initialize_backbone(target, source.state_dict())
    transferred = set(report["transferred"])
    reset = set(report["reset"])

    assert reset == {
        "adapters.0.weight",
        "embed_tokens.weight",
        "lm_head.weight",
        "output_projection.weight",
    }
    assert transferred | reset == set(target.state_dict())
    for key, value in target.state_dict().items():
        expected = source.state_dict()[key] if key in transferred else fresh[key]
        assert torch.equal(value, expected)
    assert target.lm_head.weight is target.embed_tokens.weight


def test_backbone_initialization_rejects_incomplete_state():
    model = model_with(SwiGLUSpec(16))
    state = dict(model.state_dict())
    state.pop("norm.weight")
    with pytest.raises(ValueError, match="missing norm.weight"):
        initialize_backbone(model, state)


def test_heterogeneous_head_dimensions_and_widths():
    config = ArchitectureConfig(
        (
            BlockGroup(BlockConfig(8, (StageConfig((AttentionSpec(4, 1),)),))),
            BlockGroup(BlockConfig(12, (StageConfig((AttentionSpec(6, 1),)),))),
        ),
        8,
        vocab_size=16,
        max_position_embeddings=8,
    )
    model = SpeckForCausalLM(config)
    model.init_weights()
    tokens = torch.randint(0, 16, (1, 6))
    assert torch.allclose(model(tokens), cached_logits(model, tokens), atol=1e-5)
    assert set(model.rotary) == {"4", "6"}


def test_parallel_stage_cache_matches_full_forward():
    config = ArchitectureConfig(
        (
            BlockGroup(
                BlockConfig(
                    8,
                    (
                        StageConfig(
                            (
                                AttentionSpec(4, 1, "sliding", 3),
                                GatedCausalConvSpec(8, 3),
                            )
                        ),
                    ),
                )
            ),
        ),
        8,
        vocab_size=16,
        max_position_embeddings=8,
    )
    model = SpeckForCausalLM(config)
    model.init_weights()
    tokens = torch.randint(0, 16, (1, 6))
    assert torch.allclose(model(tokens), cached_logits(model, tokens), atol=1e-5)
