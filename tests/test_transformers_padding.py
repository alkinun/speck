import pytest
import torch

from speck.transformers_padding import validate_right_padding


def test_right_padding_reports_whether_padding_is_present():
    assert not validate_right_padding(None, 2, 4)
    assert not validate_right_padding(torch.ones(2, 4, dtype=torch.bool), 2, 4)
    assert validate_right_padding(
        torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]], dtype=torch.bool), 2, 4
    )


@pytest.mark.parametrize(
    ("mask", "message"),
    [
        (torch.ones(1, 3), "match the input shape"),
        (torch.tensor([[1, 2, 0], [1, 1, 1]]), "must be binary"),
        (torch.tensor([[0, 0, 0], [1, 1, 0]]), "must start with a token"),
        (torch.tensor([[1, 0, 1], [1, 1, 0]]), "right padding only"),
    ],
)
def test_right_padding_rejects_invalid_masks(mask, message):
    with pytest.raises(ValueError, match=message):
        validate_right_padding(mask, 2, 3)
