import pytest

from speck.search.calibration import (
    calibration_report,
    frontier_calibration,
    kendall_tau_b,
    pairwise_concordance,
    spearman,
    top_k_recall,
)


def test_rank_metrics_identify_order_and_reversal():
    ordered = (1.0, 2.0, 3.0, 4.0)
    reversed_values = tuple(reversed(ordered))
    assert spearman(ordered, ordered) == pytest.approx(1.0)
    assert spearman(ordered, reversed_values) == pytest.approx(-1.0)
    assert kendall_tau_b(ordered, ordered) == pytest.approx(1.0)
    assert pairwise_concordance(ordered, reversed_values) == 0.0
    assert top_k_recall(ordered, reversed_values, 2) == 0.0


def test_calibration_report_is_deterministic_and_outcome_focused():
    predicted = {f"a{index}": float(index) for index in range(8)}
    observed = {f"a{index}": float(index) + (0.1 if index % 2 else 0) for index in range(8)}
    first = calibration_report(predicted, observed, bootstrap_samples=100, seed=7)
    second = calibration_report(predicted, observed, bootstrap_samples=100, seed=7)
    assert first == second
    assert first.spearman > 0.9
    assert first.top_k_recall == 1.0
    assert len(first.digest) == 64


def test_frontier_calibration_reports_false_positives_and_negatives():
    directions = {"quality": "minimize", "latency": "minimize"}
    predicted = {
        "a": {"quality": 1.0, "latency": 3.0},
        "b": {"quality": 2.0, "latency": 2.0},
        "c": {"quality": 3.0, "latency": 1.0},
    }
    observed = {
        "a": {"quality": 1.0, "latency": 3.0},
        "b": {"quality": 2.0, "latency": 4.0},
        "c": {"quality": 3.0, "latency": 1.0},
    }
    report = frontier_calibration(predicted, observed, directions)
    assert report.predicted == ("a", "b", "c")
    assert report.observed == ("a", "c")
    assert report.recall == 1.0
    assert report.precision == pytest.approx(2 / 3)
