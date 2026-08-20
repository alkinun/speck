"""pareto ranking and diversity-aware steady-state selection."""

import math
import random
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
        distances[ordered[0].id] = math.inf
        distances[ordered[-1].id] = math.inf
        minimum = ordered[0].objectives[name]
        maximum = ordered[-1].objectives[name]
        if maximum == minimum:
            continue
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
