import pytest
import torch

from speck.model import Config, LayerConfig, SpeckForCausalLM
from speck.search.architecture import (
    SearchSpace,
    architecture_distance,
    architecture_hash,
    kv_bytes_per_token,
    mutate,
    mutation_operators,
    novelty,
    parameter_count,
    repair,
)


def space():
    return SearchSpace(
        min_layers=2,
        max_layers=4,
        hidden_size_min=8,
        hidden_size_max=16,
        hidden_size_step=4,
        intermediate_size_min=16,
        intermediate_size_max=32,
        intermediate_size_step=8,
        kv_heads=(1, 2, 4),
        min_attention_layers=1,
        max_attention_layers=3,
    )


def config():
    return Config(
        vocab_size=32,
        layers=(
            LayerConfig(8, 16, 1),
            LayerConfig(8, 16, None),
            LayerConfig(8, 16, 1),
        ),
        head_dim=4,
        max_position_embeddings=16,
    )


def test_static_metrics_match_model():
    candidate = config()
    with torch.device("meta"):
        model = SpeckForCausalLM(candidate)
    assert parameter_count(candidate) == model.parameter_count()
    assert kv_bytes_per_token(candidate) == 32
    assert architecture_hash(candidate) == architecture_hash(candidate)


@pytest.mark.parametrize("operator", mutation_operators)
def test_every_mutation_is_deterministic_and_valid(operator):
    first = mutate(config(), space(), seed=11, operator=operator)
    second = mutate(config(), space(), seed=11, operator=operator)
    assert first == second
    assert first.mutation["operator"] == operator
    assert 2 <= len(first.config.layers) <= 4
    for layer in first.config.layers:
        assert layer.hidden_size % first.config.head_dim == 0
        if layer.num_key_value_heads is not None:
            assert (layer.hidden_size // first.config.head_dim) % layer.num_key_value_heads == 0


def test_repair_records_shape_and_attention_changes():
    candidate = Config(
        vocab_size=32,
        layers=(
            LayerConfig(12, 17, None),
            LayerConfig(12, 17, None),
        ),
        head_dim=4,
    )
    repaired, actions = repair(candidate, space())
    assert repaired.layers[0].intermediate_size == 16
    assert sum(layer.num_key_value_heads is not None for layer in repaired.layers) == 1
    assert {action["kind"] for action in actions} == {
        "repair_layer",
        "enable_attention",
    }


def test_resource_constraints_reject_candidate():
    constrained = SearchSpace(
        **{
            **space().__dict__,
            "max_parameters": parameter_count(config()) - 1,
        }
    )
    with pytest.raises(ValueError, match="parameter"):
        repair(config(), constrained)


def test_architecture_distance_and_novelty():
    candidate = config()
    changed = mutate(candidate, space(), seed=5, operator="change_ffn_width").config
    assert architecture_distance(candidate, candidate, space()) == 0
    assert architecture_distance(candidate, changed, space()) > 0
    assert novelty(candidate, [], space()) == 1
    assert novelty(candidate, [changed], space()) > 0
