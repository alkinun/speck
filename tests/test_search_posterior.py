import pytest

from speck.search.posterior import CandidatePosterior, posterior_pareto
from speck.search.protocol import ObjectiveSet, ObjectiveSpec


def objectives():
    return ObjectiveSet(
        "gpu_short",
        (
            ObjectiveSpec("quality", "minimize", "quality"),
            ObjectiveSpec("speed", "maximize", "efficiency"),
        ),
    )


def candidate(name, mean, variance=0.0):
    return CandidatePosterior(
        name,
        ("quality", "speed"),
        mean,
        ((variance, 0.0), (0.0, variance)),
    )


def test_posterior_pareto_respects_objective_directions():
    metrics = posterior_pareto(
        (
            candidate("best", (1.0, 3.0)),
            candidate("middle", (2.0, 2.0)),
            candidate("worst", (3.0, 1.0)),
        ),
        objectives(),
        samples=10,
        seed=1,
    )
    by_name = {metric.architecture_digest: metric for metric in metrics}
    assert by_name["best"].nondominated_probability == 1.0
    assert by_name["best"].expected_rank == 0.0
    assert by_name["worst"].expected_rank == 2.0


def test_posterior_pareto_preserves_uncertain_candidates():
    metrics = posterior_pareto(
        (
            candidate("known", (1.0, 2.0), 0.0),
            candidate("uncertain", (1.2, 1.8), 0.25),
        ),
        objectives(),
        samples=2_000,
        seed=2,
    )
    uncertain = next(
        metric for metric in metrics if metric.architecture_digest == "uncertain"
    )
    assert 0 < uncertain.nondominated_probability <= 1


def test_candidate_posterior_rejects_invalid_covariance():
    with pytest.raises(ValueError, match="positive semidefinite"):
        CandidatePosterior(
            "invalid",
            ("quality", "speed"),
            (1.0, 2.0),
            ((1.0, 2.0), (2.0, 1.0)),
        )
