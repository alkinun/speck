import pytest
import torch

from scripts.attention_sink_diagnostic import (
    SinkAccumulator,
    positive_integer,
    query_indices,
)


def test_query_indices_cover_requested_late_range():
    assert query_indices(10, 3, 0.5) == (4, 6, 9)
    assert query_indices(10, 1, 0.5) == (9,)


def test_sink_accumulator_reports_uniform_attention_without_enrichment():
    accumulator = SinkAccumulator(prefix_tokens=2)
    accumulator.update(torch.full((1, 2, 4), 0.25), query_position=3)
    report = accumulator.report()

    assert report["rows"] == 2
    assert report["first_token_mass"] == 0.25
    assert report["first_token_enrichment_over_uniform"] == 1.0
    assert report["prefix_mass"] == 0.5
    assert report["prefix_enrichment_over_uniform"] == 1.0
    assert report["recent_prefix_sized_mass"] == 0.5
    assert report["maximum_token_mass"] == 0.25
    assert report["normalized_entropy"] == pytest.approx(1.0)
    assert report["first_token_argmax_fraction"] == 1.0


def test_sink_accumulator_merges_rows():
    first = SinkAccumulator(prefix_tokens=1)
    second = SinkAccumulator(prefix_tokens=1)
    first.update(torch.tensor([[[0.75, 0.25]]]), query_position=1)
    second.update(torch.tensor([[[0.25, 0.75]]]), query_position=1)
    first.merge(second)
    report = first.report()
    assert report["first_token_mass"] == 0.5
    assert report["first_token_argmax_fraction"] == 0.5


@pytest.mark.parametrize("value", (0, -1, True, 1.2))
def test_positive_integer_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="positive integer"):
        positive_integer(value, "example")
