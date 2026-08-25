from types import SimpleNamespace

import pytest

from scripts.model_code_publish import prepare_code_update, validate_tokenizer_size, weight_sha256
from scripts.model_publish import (
    MODEL_FORWARD_SETUP,
    MODEL_IMPORT,
    MODEL_POSITION_CHECK,
    PADDING_DESTINATION,
)


def model_source():
    return (
        MODEL_IMPORT
        + "\nclass Model:\n"
        + "    def forward(self):\n"
        + MODEL_FORWARD_SETUP
        + "        position = 0\n"
        + "        expected_positions = torch.arange(position, position + length)\n"
        + MODEL_POSITION_CHECK
    )


def test_weight_sha256_reads_model_lfs_metadata():
    files = [
        SimpleNamespace(path="README.md", lfs=None),
        SimpleNamespace(path="model.safetensors", lfs=SimpleNamespace(sha256="abc")),
    ]

    assert weight_sha256(files) == "abc"


def test_weight_sha256_requires_lfs_weights():
    with pytest.raises(ValueError, match="no LFS metadata"):
        weight_sha256([SimpleNamespace(path="model.safetensors", lfs=None)])


def test_validate_tokenizer_size_includes_added_tokens():
    class Tokenizer:
        vocab_size = 32000

        def __len__(self):
            return 32003

    validate_tokenizer_size(Tokenizer(), 32003)


def test_prepare_code_update_writes_patched_code(tmp_path):
    source = tmp_path / "modeling_speck.py"
    output = tmp_path / "output"
    source.write_text(model_source(), encoding="utf-8")

    hashes = prepare_code_update(source, output)

    assert set(hashes) == {"modeling_speck.py", PADDING_DESTINATION}
    assert "validate_right_padding" in (output / "modeling_speck.py").read_text()
    assert (output / PADDING_DESTINATION).is_file()
