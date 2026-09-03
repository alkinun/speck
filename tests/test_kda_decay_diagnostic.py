import pytest
import torch

from scripts.kda_decay_diagnostic import DecayAccumulator, positive_integer


def test_decay_accumulator_reports_exact_moments_thresholds_and_quantiles():
    accumulator = DecayAccumulator(sample_values=8)
    accumulator.update(torch.tensor([-100.0, -20.0, -5.0, -1.0]))
    report = accumulator.report()

    assert report["count"] == 4
    assert report["mean"] == -31.5
    assert report["minimum"] == -100.0
    assert report["maximum"] == -1.0
    assert report["fractions_below"]["-5"] == 0.5
    assert report["fractions_below"]["-80"] == 0.25
    assert report["quantiles"]["0.0"] == -100.0
    assert report["quantiles"]["1.0"] == -1.0


def test_decay_accumulator_merges_scopes():
    first = DecayAccumulator(sample_values=4)
    second = DecayAccumulator(sample_values=4)
    first.update(torch.tensor([-1.0, -2.0]))
    second.update(torch.tensor([-3.0, -4.0]))
    first.merge(second)
    assert first.report()["mean"] == -2.5


@pytest.mark.parametrize("value", (0, -1, True, 1.2))
def test_positive_integer_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="positive integer"):
        positive_integer(value, "example")
