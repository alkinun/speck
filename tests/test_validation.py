import pytest

from speck.validation import positive_integer


@pytest.mark.parametrize("value", (0, -1, True, 1.5, "1", None))
def test_positive_integer_rejects_non_positive_integers(value):
    with pytest.raises(ValueError, match="example must be a positive integer"):
        positive_integer(value, "example")


def test_positive_integer_returns_valid_value():
    assert positive_integer(7, "example") == 7
