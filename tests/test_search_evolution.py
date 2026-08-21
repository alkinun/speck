import math

import pytest

from speck.model import Config, LayerConfig
from speck.search.architecture import SearchSpace
from speck.search.evolution import (
    EvaluatedCandidate,
    OperatorOutcome,
    aggregate_trials,
    crowding_distance,
    dominates,
    nondominated_sort,
    operator_probabilities,
    select_parent,
    select_survivors,
)


objectives = ("quality", "latency")


def space():
    return SearchSpace(
        min_layers=1,
        max_layers=2,
        hidden_size_min=8,
        hidden_size_max=12,
        hidden_size_step=4,
        intermediate_size_min=16,
        intermediate_size_max=24,
        intermediate_size_step=8,
        kv_heads=(1, 2),
    )


def candidate(candidate_id, quality, latency, hidden=8, intermediate=16):
    config = Config(
        vocab_size=16,
        layers=(LayerConfig(hidden, intermediate, 1),),
        head_dim=4,
    )
    return EvaluatedCandidate(
        candidate_id, config, {"quality": quality, "latency": latency}
    )


def test_dominance_and_fronts():
    first = candidate(1, 1, 2)
    second = candidate(2, 2, 3)
    tradeoff = candidate(3, 3, 1)
    assert dominates(first, second, objectives)
    assert not dominates(first, tradeoff, objectives)
    fronts = nondominated_sort((first, second, tradeoff), objectives)
    assert {value.id for value in fronts[0]} == {1, 3}
    assert {value.id for value in fronts[1]} == {2}


def test_crowding_and_diversity_survival():
    candidates = (
        candidate(1, 1, 4),
        candidate(2, 2, 3, hidden=12),
        candidate(3, 3, 2, intermediate=24),
        candidate(4, 4, 1),
    )
    crowding = crowding_distance(candidates, objectives)
    assert math.isinf(crowding[1])
    assert math.isinf(crowding[4])
    selected, metrics, frontier = select_survivors(
        candidates, 3, objectives, space()
    )
    assert len(selected) == 3
    assert set(frontier) == {1, 2, 3, 4}
    assert all(candidate_id in metrics for candidate_id in selected)
    parent = select_parent(
        [value for value in candidates if value.id in selected], metrics, seed=9
    )
    assert parent.id in selected


def test_invalid_objective_is_rejected():
    invalid = candidate(1, 1, math.nan)
    with pytest.raises(ValueError, match="invalid objective"):
        nondominated_sort((invalid,), objectives)


def test_constant_objective_does_not_create_boundaries():
    candidates = tuple(candidate(index, 1, 1) for index in range(1, 4))
    assert crowding_distance(candidates, objectives) == {1: 0.0, 2: 0.0, 3: 0.0}


def test_trial_aggregation_has_confidence_and_static_values():
    estimates = aggregate_trials(
        (
            {"objectives": {"quality": 2.0, "latency": 4.0}},
            {"objectives": {"quality": 2.2, "latency": 4.2}},
        ),
        {"memory": 100},
        ("quality", "latency", "memory"),
    )
    assert estimates["quality"].mean == pytest.approx(2.1)
    assert estimates["quality"].lower < 2.1 < estimates["quality"].upper
    assert estimates["memory"].n == 0
    assert estimates["memory"].lower == estimates["memory"].upper == 100


def test_operator_probabilities_adapt_with_an_exploration_floor():
    probabilities = operator_probabilities(
        (
            OperatorOutcome("remove_layer", True),
            OperatorOutcome("remove_layer", True),
            OperatorOutcome("change_hidden_size", False),
        ),
        ("remove_layer", "change_hidden_size"),
        probability_floor=0.1,
    )
    assert sum(probabilities.values()) == pytest.approx(1)
    assert probabilities["remove_layer"] > probabilities["change_hidden_size"]
    assert min(probabilities.values()) >= 0.1
