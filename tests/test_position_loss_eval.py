import pytest
import torch

from scripts.position_loss_eval import (
    evaluate_position_loss,
    position_ranges,
    positive_integer,
    summarize_sums,
)


class LossModel(torch.nn.Module):
    def __init__(self, batches):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.batches = iter(batches)

    def forward(self, _inputs, _targets, loss_reduction):
        assert loss_reduction == "none"
        return next(self.batches).flatten()


def test_position_ranges_cover_nondivisible_sequence_exactly():
    ranges = position_ranges(10, 3)
    assert ranges == ((0, 3), (3, 6), (6, 10))
    assert [index for start, end in ranges for index in range(start, end)] == list(range(10))


@pytest.mark.parametrize("value", (0, -1, True, 1.5))
def test_positive_integer_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="positive integer"):
        positive_integer(value, "example")


def test_summarize_sums_omits_empty_bins():
    assert summarize_sums([3.0, 0.0], [2, 0], ((0, 2), (2, 4))) == [
        {"start": 0, "end": 2, "tokens": 2, "loss": 1.5}
    ]


def test_position_loss_aggregates_bins_trailing_region_and_sources():
    batches = [
        torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        torch.tensor([[5.0, 6.0, 7.0, 8.0]]),
    ]
    loader = iter(
        [
            (
                torch.zeros(1, 4, dtype=torch.long),
                torch.zeros(1, 4, dtype=torch.long),
                {"selected_source": "a"},
            ),
            (
                torch.zeros(1, 4, dtype=torch.long),
                torch.zeros(1, 4, dtype=torch.long),
                {"selected_source": "b"},
            ),
        ]
    )
    result = evaluate_position_loss(
        LossModel(batches), loader, 2, ("a", "b"), sequence_length=4, bins=2, trailing_tokens=2
    )

    assert result["loss"] == 4.5
    assert [item["loss"] for item in result["position_bins"]] == [3.5, 5.5]
    assert result["trailing_loss"] == 5.5
    assert [item["loss"] for item in result["source_position_bins"]["a"]] == [1.5, 3.5]
    assert [item["loss"] for item in result["source_position_bins"]["b"]] == [5.5, 7.5]
    assert result["source_trailing_loss"] == {"a": 3.5, "b": 7.5}
