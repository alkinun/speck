import hashlib
import json
from pathlib import Path

import pytest

import speck.data.stock_tokens as stock
from speck.provenance.io import file_sha256


class ByteTokenizer:
    vocab_size = 256

    def __init__(self, path):
        self.path = path

    def fingerprint(self):
        return file_sha256(self.path)

    def encode_batch(self, texts, **kwargs):
        assert kwargs == {"bos": True, "eos": True}
        return [[1, *text.encode(), 2] for text in texts]


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(stock, "Tokenizer", ByteTokenizer)
    texts = [
        "A short document.",
        "Unicode λ and English in the middle.",
        "Final document across shard boundaries.",
    ]
    rows = [
        {
            "text": text,
            "content_id": str(i),
            "released_content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
        for i, text in enumerate(texts)
    ]
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    model = tmp_path / "model"
    model.write_text("fixture")
    plan = {
        "output_directory": str(tmp_path / "tokens"),
        "input": {"path": str(source), "sha256": file_sha256(source)},
        "tokenizer": {"path": str(model), "sha256": file_sha256(model)},
        "expected_documents": len(rows),
        "expected_tokens": sum(len(text.encode()) + 2 for text in texts),
        "shard_tokens": 11,
    }
    return plan, texts


def test_document_offsets_reconstruct_cross_shard_tokens_and_reopen(tmp_path, monkeypatch):
    plan, texts = fixture(tmp_path, monkeypatch)
    result = stock.tokenize_stock(plan)
    output = Path(plan["output_directory"])
    manifest = stock.verify_token_stock(output, plan)
    rows = [json.loads(raw) for raw in (output / "documents.jsonl").read_text().splitlines()]
    for row, text in zip(rows, texts):
        assert stock._read_span(
            output, manifest["shards"], row["token_start"], row["token_count"]
        ) == [1, *text.encode(), 2]
    assert len(manifest["shards"]) > len(rows)
    reopened = stock.tokenize_stock(plan)
    assert reopened["reused"] is True
    assert reopened["manifest"] == result["manifest"]
    assert manifest["training_authority"] is False


@pytest.mark.parametrize("payload", ["documents.jsonl", "tokens_00000.bin"])
def test_changed_payload_cannot_reopen(tmp_path, monkeypatch, payload):
    plan, _ = fixture(tmp_path, monkeypatch)
    stock.tokenize_stock(plan)
    (Path(plan["output_directory"]) / payload).write_bytes(b"changed")
    with pytest.raises(ValueError, match="payload identity"):
        stock.tokenize_stock(plan)


def test_count_mismatch_stays_unpublished_and_partial_build_is_preserved(tmp_path, monkeypatch):
    plan, _ = fixture(tmp_path, monkeypatch)
    plan["expected_tokens"] += 1
    with pytest.raises(ValueError, match="counts differ"):
        stock.tokenize_stock(plan)
    assert not Path(plan["output_directory"]).exists()
    assert Path(plan["output_directory"] + ".building/plan.json").exists()
    with pytest.raises(FileExistsError):
        stock.tokenize_stock(plan)
