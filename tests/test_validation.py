import pytest

from speck.validation import positive_integer, require_keys


@pytest.mark.parametrize("value", (0, -1, True, 1.5, "1", None))
def test_positive_integer_rejects_non_positive_integers(value):
    with pytest.raises(ValueError, match="example must be a positive integer"):
        positive_integer(value, "example")


def test_positive_integer_returns_valid_value():
    assert positive_integer(7, "example") == 7


def test_require_keys_reports_sorted_missing_fields():
    with pytest.raises(ValueError, match="record is missing required fields: a, c"):
        require_keys({"b": 2}, {"c", "b", "a"}, "record")


def test_require_keys_accepts_additional_fields():
    assert require_keys({"a": 1, "extra": 2}, {"a"}, "record") is None
