"""Ladder corpora take training only from train families and validation only from development."""

import hashlib
import json

import pytest

from speck.data import ladder
from speck.provenance.io import file_sha256


class Tokenizer:
    vocab_size = 32000
    bos_id = 1
    eos_id = 2

    def encode_batch(self, texts, bos=False, eos=False):
        return [[1, *(byte + 3 for byte in text.encode()), 2] for text in texts]

    def fingerprint(self):
        return "test-tokenizer"


BUCKETS = ["train", "development", "final", "quarantine"]


def _source(tmp_path, name):
    texts = [f"{name} document {index} " + "x" * 20 for index in range(40)]
    text = tmp_path / f"{name}.jsonl"
    rows = []
    with text.open("w") as handle:
        for index, value in enumerate(texts):
            digest = hashlib.sha256(value.encode()).hexdigest()
            handle.write(json.dumps({"text": value, "released_content_sha256": digest}) + "\n")
            rows.append(
                {
                    "source": name,
                    "released_content_sha256": digest,
                    "candidate_partition": BUCKETS[index % 4],
                }
            )
    return {"id": name, "text": {"path": str(text), "sha256": file_sha256(text)}}, rows


def _experiment(tmp_path, name, weights, inputs):
    directory = tmp_path / name
    directory.mkdir()
    sources = [
        {
            "id": source,
            "repo": f"local/{source}",
            "revision": None,
            "tree_path": "",
            "content_column": "text",
            "filters": {},
        }
        for source in weights
    ]
    data = {
        "sources": sources,
        "mixture": {"phases": [{"end_tokens": 200, "weights": weights}]},
        "requested_train_tokens": 200,
        "validation_tokens_per_source": 90,
        "validation_fraction": 0.5,
        "filtering": {"min_chars": 0, "max_chars": 10_000},
        "dedup": {
            "normalization": "NFKC+lower+whitespace",
            "hash": "blake2b-128",
            "scope": "global",
        },
        "shards": {"tokens": 64, "maximum_loader_microbatch_tokens": 0},
        "seed": 3,
        "output_dir": str(tmp_path / f"{name}-packed"),
    }
    (directory / "data.json").write_text(json.dumps(data))
    (directory / "tokenizer.json").write_text("{}")
    return ladder.prepare(directory, inputs), tmp_path / f"{name}-packed"


def _records(packed):
    manifest = json.loads((packed / "manifest.json").read_text())
    records = {}
    for source in manifest["sources"]:
        index = packed / source["document_index"]["path"]
        records[source["id"]] = [json.loads(line) for line in index.read_text().splitlines()]
    return records


def test_splits_follow_family_buckets_and_validation_ignores_the_mixture(tmp_path, monkeypatch):
    monkeypatch.setattr(ladder, "get_tokenizer", lambda **_: Tokenizer())
    (web, web_rows), (code, code_rows) = _source(tmp_path, "web"), _source(tmp_path, "code")
    partitions = tmp_path / "partitions.jsonl"
    partitions.write_text("".join(json.dumps(row) + "\n" for row in web_rows + code_rows))
    inputs = tmp_path / "inputs.json"
    inputs.write_text(
        json.dumps(
            {
                "format": "speck_ladder_inputs",
                "format_version": 1,
                "partitions": {"path": str(partitions), "sha256": file_sha256(partitions)},
                "sources": [web, code],
            }
        )
    )
    bucket = {
        row["released_content_sha256"]: row["candidate_partition"] for row in web_rows + code_rows
    }
    receipt, first = _experiment(tmp_path, "mostly-web", {"web": 80, "code": 20}, inputs)
    _, second = _experiment(tmp_path, "mostly-code", {"web": 20, "code": 80}, inputs)

    for packed in (first, second):
        for records in _records(packed).values():
            for record in records:
                expected = "train" if record["split"] == "train" else "development"
                assert bucket[record["content_hash"]] == expected
    validation = [
        {r["content_hash"] for r in records if r["split"] == "val"}
        for records in (_records(first)["web"], _records(second)["web"])
    ]
    assert validation[0] and validation[0] == validation[1]
    assert receipt["training_admitted"] is False


def test_a_changed_input_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(ladder, "get_tokenizer", lambda **_: Tokenizer())
    web, rows = _source(tmp_path, "web")
    partitions = tmp_path / "partitions.jsonl"
    partitions.write_text("".join(json.dumps(row) + "\n" for row in rows))
    inputs = tmp_path / "inputs.json"
    inputs.write_text(
        json.dumps(
            {
                "format": "speck_ladder_inputs",
                "format_version": 1,
                "partitions": {"path": str(partitions), "sha256": file_sha256(partitions)},
                "sources": [web],
            }
        )
    )
    with open(web["text"]["path"], "a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="web text changed"):
        _experiment(tmp_path, "changed", {"web": 100}, inputs)
