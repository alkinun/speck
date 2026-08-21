"""pareto ranking and diversity-aware steady-state selection."""

import math
import random
import statistics
from dataclasses import dataclass

from speck.model import Config
from speck.search.architecture import SearchSpace, novelty


@dataclass(frozen=True)
class EvaluatedCandidate:
    id: int
    config: Config
    objectives: dict[str, float]


@dataclass(frozen=True)
class SelectionMetrics:
    rank: int
    crowding: float
    novelty: float


@dataclass(frozen=True)
class ObjectiveEstimate:
    n: int
    mean: float
    stdev: float
    lower: float
    upper: float


@dataclass(frozen=True)
class EvaluatedArchitecture:
    id: int
    architecture_hash: str
    config: Config
    objectives: dict[str, ObjectiveEstimate]


@dataclass(frozen=True)
class OperatorOutcome:
    operator: str
    success: bool


def aggregate_trials(results, static_objectives, objective_names, confidence_z=1.645):
    estimates = {}
    for name in objective_names:
        if name in static_objectives:
            value = float(static_objectives[name])
            estimates[name] = ObjectiveEstimate(0, value, 0.0, value, value)
            continue
        values = [float(result["objectives"][name]) for result in results]
        if not values or any(not math.isfinite(value) for value in values):
            raise ValueError(f"cannot aggregate objective: {name}")
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        if name.startswith("quality."):
            floor = 0.03
        elif name.startswith(("prefill.", "decode.")):
            floor = abs(mean) * 0.01
        elif name.startswith("memory.inference_peak"):
            floor = abs(mean) * 0.005
        else:
            floor = 0.0
        standard_error = max(
            stdev / math.sqrt(len(values)) if len(values) > 1 else 0.0,
            floor / math.sqrt(len(values)),
        )
        half_width = confidence_z * standard_error
        estimates[name] = ObjectiveEstimate(
            len(values),
            mean,
            stdev,
            max(0.0, mean - half_width),
            mean + half_width,
        )
    return estimates


def estimated_candidates(candidates, bound="mean"):
    if bound not in {"mean", "lower", "upper"}:
        raise ValueError("objective estimate bound must be mean, lower, or upper")
    return tuple(
        EvaluatedCandidate(
            candidate.id,
            candidate.config,
            {
                name: getattr(estimate, bound)
                for name, estimate in candidate.objectives.items()
            },
        )
        for candidate in candidates
    )


def operator_probabilities(
    outcomes,
    operators,
    prior_success=1.0,
    prior_failure=1.0,
    probability_floor=0.04,
):
    if not operators:
        raise ValueError("at least one operator is required")
    if probability_floor * len(operators) >= 1:
        raise ValueError("operator probability floor is too large")
    counts = {operator: [prior_success, prior_failure] for operator in operators}
    for outcome in outcomes:
        if outcome.operator in counts:
            counts[outcome.operator][0 if outcome.success else 1] += 1
    rates = {
        operator: success / (success + failure)
        for operator, (success, failure) in counts.items()
    }
    available = 1 - probability_floor * len(operators)
    total = sum(rates.values())
    return {
        operator: probability_floor + available * rates[operator] / total
        for operator in operators
    }


def _validate(candidates, objective_names):
    if not candidates:
        return
    for candidate in candidates:
        missing = set(objective_names) - set(candidate.objectives)
        if missing:
            raise ValueError(f"candidate is missing objectives: {', '.join(sorted(missing))}")
        for name in objective_names:
            value = candidate.objectives[name]
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"candidate has invalid objective: {name}")


def dominates(left, right, objective_names):
    _validate((left, right), objective_names)
    no_worse = all(
        left.objectives[name] <= right.objectives[name] for name in objective_names
    )
    better = any(
        left.objectives[name] < right.objectives[name] for name in objective_names
    )
    return no_worse and better


def nondominated_sort(candidates, objective_names):
    candidates = tuple(candidates)
    _validate(candidates, objective_names)
    dominated_by = {candidate.id: 0 for candidate in candidates}
    dominates_ids = {candidate.id: [] for candidate in candidates}
    by_id = {candidate.id: candidate for candidate in candidates}
    for index, left in enumerate(candidates):
        for right in candidates[index + 1:]:
            if dominates(left, right, objective_names):
                dominates_ids[left.id].append(right.id)
                dominated_by[right.id] += 1
            elif dominates(right, left, objective_names):
                dominates_ids[right.id].append(left.id)
                dominated_by[left.id] += 1

    fronts = []
    current = sorted(
        candidate_id for candidate_id, count in dominated_by.items() if count == 0
    )
    while current:
        fronts.append(tuple(by_id[candidate_id] for candidate_id in current))
        following = []
        for candidate_id in current:
            for dominated_id in dominates_ids[candidate_id]:
                dominated_by[dominated_id] -= 1
                if dominated_by[dominated_id] == 0:
                    following.append(dominated_id)
        current = sorted(following)
    return tuple(fronts)


def crowding_distance(candidates, objective_names):
    candidates = tuple(candidates)
    _validate(candidates, objective_names)
    distances = {candidate.id: 0.0 for candidate in candidates}
    if len(candidates) <= 2:
        return {candidate.id: math.inf for candidate in candidates}
    for name in objective_names:
        ordered = sorted(candidates, key=lambda candidate: (candidate.objectives[name], candidate.id))
        minimum = ordered[0].objectives[name]
        maximum = ordered[-1].objectives[name]
        if maximum == minimum:
            continue
        distances[ordered[0].id] = math.inf
        distances[ordered[-1].id] = math.inf
        for index in range(1, len(ordered) - 1):
            if math.isinf(distances[ordered[index].id]):
                continue
            previous = ordered[index - 1].objectives[name]
            following = ordered[index + 1].objectives[name]
            distances[ordered[index].id] += (following - previous) / (maximum - minimum)
    return distances


def selection_metrics(candidates, objective_names, space):
    candidates = tuple(candidates)
    fronts = nondominated_sort(candidates, objective_names)
    metrics = {}
    configs = [candidate.config for candidate in candidates]
    for rank, front in enumerate(fronts):
        crowding = crowding_distance(front, objective_names)
        for candidate in front:
            metrics[candidate.id] = SelectionMetrics(
                rank=rank,
                crowding=crowding[candidate.id],
                novelty=novelty(
                    candidate.config,
                    [config for config in configs if config is not candidate.config],
                    space,
                ),
            )
    return metrics, fronts


def select_survivors(candidates, size, objective_names, space):
    candidates = tuple(candidates)
    if size < 1:
        raise ValueError("population size must be positive")
    metrics, fronts = selection_metrics(candidates, objective_names, space)
    selected = []
    for front in fronts:
        remaining = size - len(selected)
        if remaining <= 0:
            break
        if len(front) <= remaining:
            selected.extend(candidate.id for candidate in front)
            continue
        pool = list(front)
        use_crowding = True
        while len(selected) < size:
            if use_crowding:
                chosen = max(
                    pool,
                    key=lambda candidate: (
                        metrics[candidate.id].crowding,
                        metrics[candidate.id].novelty,
                        -candidate.id,
                    ),
                )
            else:
                chosen = max(
                    pool,
                    key=lambda candidate: (
                        metrics[candidate.id].novelty,
                        metrics[candidate.id].crowding,
                        -candidate.id,
                    ),
                )
            selected.append(chosen.id)
            pool.remove(chosen)
            use_crowding = not use_crowding
        break
    return tuple(selected), metrics, tuple(candidate.id for candidate in fronts[0])


def select_parent(candidates, metrics, seed):
    candidates = tuple(candidates)
    if not candidates:
        raise ValueError("cannot select a parent from an empty population")
    if len(candidates) == 1:
        return candidates[0]
    rng = random.Random(seed)
    left, right = rng.sample(candidates, 2)
    left_metrics = metrics[left.id]
    right_metrics = metrics[right.id]
    if left_metrics.rank != right_metrics.rank:
        return left if left_metrics.rank < right_metrics.rank else right
    if rng.random() < 0.5:
        left_key = (left_metrics.crowding, left_metrics.novelty, -left.id)
        right_key = (right_metrics.crowding, right_metrics.novelty, -right.id)
    else:
        left_key = (left_metrics.novelty, left_metrics.crowding, -left.id)
        right_key = (right_metrics.novelty, right_metrics.crowding, -right.id)
    return left if left_key >= right_key else right
