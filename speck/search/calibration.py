"""out-of-sample rank and frontier calibration reports."""

import math
from dataclasses import asdict, dataclass

import numpy as np

from speck.search.protocol import content_digest


def _rank(values):
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return tuple(ranks)


def _correlation(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left -= left.mean()
    right -= right.mean()
    denominator = math.sqrt(float(left @ left) * float(right @ right))
    return float(left @ right / denominator) if denominator else 0.0


def spearman(left, right):
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("rank correlation needs equal samples with at least two values")
    return _correlation(_rank(left), _rank(right))


def kendall_tau_b(left, right):
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("rank correlation needs equal samples with at least two values")
    concordant = discordant = left_ties = right_ties = 0
    for first in range(len(left)):
        for second in range(first + 1, len(left)):
            left_delta = left[first] - left[second]
            right_delta = right[first] - right[second]
            if left_delta == 0 and right_delta == 0:
                continue
            if left_delta == 0:
                left_ties += 1
            elif right_delta == 0:
                right_ties += 1
            elif left_delta * right_delta > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + left_ties)
        * (concordant + discordant + right_ties)
    )
    return (concordant - discordant) / denominator if denominator else 0.0


def top_k_recall(predicted, observed, count):
    if len(predicted) != len(observed) or not 1 <= count <= len(predicted):
        raise ValueError("invalid top k comparison")
    predicted_top = set(sorted(range(len(predicted)), key=lambda index: predicted[index])[:count])
    observed_top = set(sorted(range(len(observed)), key=lambda index: observed[index])[:count])
    return len(predicted_top & observed_top) / count


def pairwise_concordance(predicted, observed):
    if len(predicted) != len(observed) or len(predicted) < 2:
        raise ValueError("pairwise concordance needs equal nontrivial samples")
    correct = compared = 0
    for first in range(len(predicted)):
        for second in range(first + 1, len(predicted)):
            predicted_delta = predicted[first] - predicted[second]
            observed_delta = observed[first] - observed[second]
            if predicted_delta == 0 or observed_delta == 0:
                continue
            compared += 1
            correct += predicted_delta * observed_delta > 0
    return correct / compared if compared else 0.0


def pareto_front(points, directions):
    names = tuple(points)
    if not names:
        return ()
    objectives = tuple(directions)
    for values in points.values():
        if set(values) != set(objectives):
            raise ValueError("pareto points do not match objective directions")

    def normalized(name, objective):
        value = points[name][objective]
        return value if directions[objective] == "minimize" else -value

    frontier = []
    for candidate in names:
        dominated = False
        for other in names:
            if candidate == other:
                continue
            no_worse = all(
                normalized(other, objective) <= normalized(candidate, objective)
                for objective in objectives
            )
            better = any(
                normalized(other, objective) < normalized(candidate, objective)
                for objective in objectives
            )
            if no_worse and better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return tuple(sorted(frontier))


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float


@dataclass(frozen=True)
class CalibrationReport:
    architectures: tuple[str, ...]
    spearman: float
    spearman_interval: Interval
    kendall_tau_b: float
    kendall_interval: Interval
    pairwise_concordance: float
    top_k_recall: float
    mean_absolute_error: float
    bootstrap_samples: int

    @property
    def digest(self):
        return content_digest(self)

    def export(self):
        return asdict(self)


@dataclass(frozen=True)
class FrontierCalibration:
    predicted: tuple[str, ...]
    observed: tuple[str, ...]
    recall: float
    precision: float


def _interval(values, coverage=0.9):
    tail = (1 - coverage) / 2
    return Interval(
        float(np.quantile(values, tail)),
        float(np.quantile(values, 1 - tail)),
    )


def calibration_report(predicted, observed, top_k=None, bootstrap_samples=1_000, seed=0):
    if set(predicted) != set(observed) or len(predicted) < 2:
        raise ValueError("calibration inputs must contain the same architectures")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap sample count must be positive")
    architectures = tuple(sorted(predicted))
    predicted_values = tuple(float(predicted[name]) for name in architectures)
    observed_values = tuple(float(observed[name]) for name in architectures)
    if any(not math.isfinite(value) for value in predicted_values + observed_values):
        raise ValueError("calibration values must be finite")
    top_k = top_k or max(1, math.ceil(len(architectures) / 4))
    rng = np.random.default_rng(seed)
    bootstrap_spearman = []
    bootstrap_kendall = []
    while len(bootstrap_spearman) < bootstrap_samples:
        indices = rng.integers(0, len(architectures), len(architectures))
        left = tuple(predicted_values[index] for index in indices)
        right = tuple(observed_values[index] for index in indices)
        bootstrap_spearman.append(spearman(left, right))
        bootstrap_kendall.append(kendall_tau_b(left, right))
    return CalibrationReport(
        architectures=architectures,
        spearman=spearman(predicted_values, observed_values),
        spearman_interval=_interval(bootstrap_spearman),
        kendall_tau_b=kendall_tau_b(predicted_values, observed_values),
        kendall_interval=_interval(bootstrap_kendall),
        pairwise_concordance=pairwise_concordance(
            predicted_values,
            observed_values,
        ),
        top_k_recall=top_k_recall(predicted_values, observed_values, top_k),
        mean_absolute_error=float(
            np.mean(np.abs(np.asarray(predicted_values) - observed_values))
        ),
        bootstrap_samples=bootstrap_samples,
    )


def frontier_calibration(predicted, observed, directions):
    predicted_frontier = pareto_front(predicted, directions)
    observed_frontier = pareto_front(observed, directions)
    overlap = set(predicted_frontier) & set(observed_frontier)
    return FrontierCalibration(
        predicted_frontier,
        observed_frontier,
        len(overlap) / len(observed_frontier),
        len(overlap) / len(predicted_frontier),
    )
