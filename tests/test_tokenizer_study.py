import pytest
from sentencepiece import sentencepiece_model_pb2

from speck.tokenizer_study import (
    _paragraph_chunks,
    _summarize_documents,
    _validate_train_spec,
)


def test_paragraph_chunks_preserve_every_character_and_byte_limit():
    text = "first paragraph\n\n" + "λ" * 400 + "\nlast"
    chunks = _paragraph_chunks(text, 128)

    assert "".join(chunks) == text
    assert all(0 < len(chunk.encode()) <= 128 for chunk in chunks)


def test_document_summary_is_byte_weighted_with_document_tails():
    documents = [
        {"utf8_bytes": 100, "tokens": 25},
        {"utf8_bytes": 300, "tokens": 150},
    ]
    result = _summarize_documents(documents)

    assert result["tokens_per_kib"] == 1024 * 175 / 400
    assert result["document_tokens_per_kib_p50"] == 256
    assert result["document_tokens_per_kib_p99"] == 512


def test_train_spec_allows_only_bounded_deterministic_trainer_overrides():
    spec = {
        "format": "speck_tokenizer_study_train_spec",
        "format_version": 1,
        "id": "profile",
        "sample_directory": "/sample",
        "output_directory": "/output",
        "source_byte_targets": {"code/source": 100},
        "model_type": "bpe",
        "vocab_size": 32768,
        "add_dummy_prefix": True,
        "training_unit": "paragraph_4k",
        "maximum_chunk_bytes": 4096,
        "trainer_overrides": {
            "allow_whitespace_only_pieces": True,
            "character_coverage": 0.99995,
            "max_sentence_length": 4192,
            "train_extremely_large_corpus": False,
        },
    }

    assert _validate_train_spec(spec) == spec
    spec["trainer_overrides"]["shuffle_input_sentence"] = True
    with pytest.raises(ValueError, match="invalid tokenizer study train spec"):
        _validate_train_spec(spec)


def test_piece_structure_counts_multi_space_usage(tmp_path):
    from speck.tokenizer_study import _piece_structure

    model = sentencepiece_model_pb2.ModelProto()
    for piece, piece_type in (("<unk>", 2), ("▁", 1), ("▁▁▁▁", 1), ("word", 1)):
        value = model.pieces.add()
        value.piece = piece
        value.type = piece_type
    path = tmp_path / "tokenizer.model"
    path.write_bytes(model.SerializeToString())

    result = _piece_structure(path, {1: 10, 2: 3, 3: 20})

    assert result["whitespace_only_pieces"] == 2
    assert result["multi_space_pieces"] == 1
    assert result["whitespace_piece_token_occurrences"] == 13
    assert result["multi_space_piece_token_occurrences"] == 3
