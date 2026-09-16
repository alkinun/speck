import pytest

from speck.data.long_document_structure import inspect_structure


class Recorder:
    def encode(self, text, *, bos, eos):
        assert bos and eos
        self.text = text
        return [1, *text, 2]


def test_padding_diagnostic_preserves_lines_content_and_original_text():
    text = "A|" + " " * 100 + "|B\n\nC\t\tD"
    tokenizer = Recorder()
    result = inspect_structure(text, tokenizer)
    assert tokenizer.text == "A| |B\n\nC D"
    assert text.count(" ") == 100
    assert result["longest_horizontal_whitespace_run"] == 100
    assert result["whitespace_characters"] == 104
    assert result["pipe_characters"] == 2 and result["lines"] == 3
    assert result["horizontal_whitespace_collapsed_tokens_including_bos_eos"] == 12
    assert not result["collapsed_length_plus_lookahead"]["32768"]


@pytest.mark.parametrize("text", ["", "a" * 2_000_001, None])
def test_structure_review_is_bounded(text):
    with pytest.raises(ValueError):
        inspect_structure(text, Recorder())
