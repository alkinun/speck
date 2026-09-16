import pytest

from speck.data.long_document_probe import probe_rows


class Tokenizer:
    def encode(self, text, *, bos, eos):
        assert bos and eos
        return [0] * (len(text) + 2)


def test_bounded_largest_sample_preserves_row_identity_without_text_publication():
    rows = [
        {"text": "a" * 20, "id": "first", "in_language": "en"},
        {"text": "b" * 50, "id": "over-cap", "in_language": "en"},
        {"text": "c" * 20, "id": "tie", "in_language": "en"},
        {"text": "d" * 30, "id": "longer", "in_language": "en"},
        {"text": "e" * 31, "id": "other-language", "in_language": "fr"},
        {"text": None, "id": "missing", "in_language": "en"},
    ]
    result = probe_rows(rows, Tokenizer(), minimum_chars=15, maximum_chars=40, sample_size=2)
    assert result["physical_rows"] == 6
    assert result["english_rows_at_least_minimum_chars"] == 4
    assert result["english_rows_above_probe_character_cap"] == 1
    assert [r["upstream_id"] for r in result["sample"]] == ["longer", "first"]
    assert [r["physical_row"] for r in result["sample"]] == [3, 0]
    assert all("text" not in r for r in result["sample"])
    assert result["qualified_stock_tokens"] is None


def test_context_candidate_still_needs_shifted_label_lookahead():
    rows = [{"text": "x" * 32766, "id": "exact-32k", "in_language": "en"}]
    result = probe_rows(rows, Tokenizer(), minimum_chars=1, maximum_chars=40000, sample_size=1)
    assert result["sample"][0]["tokens_including_bos_eos"] == 32768
    assert not result["sample"][0]["fits_intact_context_plus_lookahead"]["32768"]


def test_invalid_probe_cannot_allocate_unbounded_sample():
    with pytest.raises(ValueError):
        probe_rows([], Tokenizer(), minimum_chars=1, maximum_chars=40000, sample_size=100000)
