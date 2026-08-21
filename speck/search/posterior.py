"""joint posterior pareto estimates for dynamic search decisions."""

import math
from dataclasses import dataclass

import numpy as np

from speck.search.protocol import ObjectiveSet


@dataclass(frozen=True)
class CandidatePosterior:
    architecture_digest: str
    objective_names: tuple[str, ...]
    mean: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]

    def __post_init__(self):
        size = len(self.objective_names)
        if not self.architecture_digest or size == 0:
            raise ValueError("candidate posterior identity cannot be empty")
        if len(set(self.objective_names)) != size:
            raise ValueError("candidate posterior objectives must be unique")
        if len(self.mean) != size or len(self.covariance) != size:
            raise ValueError("candidate posterior dimensions do not match")
        if any(len(row) != size for row in self.covariance):
            raise ValueError("candidate posterior covariance must be square")
        values = self.mean + tuple(value for row in self.covariance for value in row)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("candidate posterior values must be finite")
        covariance = np.asarray(self.covariance, dtype=np.float64)
        if not np.allclose(covariance, covariance.T):
            raise ValueError("candidate posterior covariance must be symmetric")
        if np.linalg.eigvalsh(covariance).min() < -1e-10:
            raise ValueError("candidate posterior covariance must be positive semidefinite")


@dataclass(frozen=True)
class PosteriorParetoMetric:
    architecture_digest: str
    nondominated_probability: float
    expected_rank: float


def _pareto_ranks(values):
    count = len(values)
    dominates = [set() for _ in range(count)]
    dominated_by = [0] * count
    for left in range(count):
        for right in range(left + 1, count):
            left_dominates = np.all(values[left] <= values[right]) and np.any(
                values[left] < values[right]
            )
            right_dominates = np.all(values[right] <= values[left]) and np.any(
                values[right] < values[left]
            )
            if left_dominates:
                dominates[left].add(right)
                dominated_by[right] += 1
            elif right_dominates:
                dominates[right].add(left)
                dominated_by[left] += 1
    ranks = [-1] * count
    front = [index for index, value in enumerate(dominated_by) if value == 0]
    rank = 0
    while front:
        following = []
        for index in front:
            ranks[index] = rank
            for dominated in dominates[index]:
                dominated_by[dominated] -= 1
                if dominated_by[dominated] == 0:
                    following.append(dominated)
        front = following
        rank += 1
    return tuple(ranks)


def posterior_pareto(candidates, objectives, samples=2_000, seed=0):
    if not isinstance(objectives, ObjectiveSet):
        raise TypeError("posterior pareto requires an objective set")
    if not candidates or samples < 1:
        raise ValueError("posterior pareto needs candidates and samples")
    names = tuple(objective.name for objective in objectives.selection)
    directions = np.asarray(
        [1 if objective.direction == "minimize" else -1 for objective in objectives.selection],
        dtype=np.float64,
    )
    if len({candidate.architecture_digest for candidate in candidates}) != len(candidates):
        raise ValueError("candidate posterior identities must be unique")
    if any(candidate.objective_names != names for candidate in candidates):
        raise ValueError("candidate posteriors do not match the objective set")
    rng = np.random.default_rng(seed)
    rank_totals = np.zeros(len(candidates), dtype=np.float64)
    nondominated = np.zeros(len(candidates), dtype=np.int64)
    for _ in range(samples):
        draw = np.stack(
            [
                rng.multivariate_normal(candidate.mean, candidate.covariance)
                for candidate in candidates
            ]
        )
        ranks = np.asarray(_pareto_ranks(draw * directions), dtype=np.int64)
        rank_totals += ranks
        nondominated += ranks == 0
    return tuple(
        PosteriorParetoMetric(
            candidate.architecture_digest,
            float(nondominated[index] / samples),
            float(rank_totals[index] / samples),
        )
        for index, candidate in enumerate(candidates)
    )
