"""Ladder corpora take training only from train families and validation only from development."""

import hashlib
import json

import numpy as np
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


def _experiment(tmp_path, name, weights, inputs, *, tokens=200, passes=None):
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
            **({"passes": passes[source]} if source in (passes or {}) else {}),
        }
        for source in weights
    ]
    data = {
        "sources": sources,
        "mixture": {"phases": [{"end_tokens": tokens, "weights": weights}]},
        "requested_train_tokens": tokens,
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


def _inputs(tmp_path, *names):
    sources, rows = [], []
    for name in names:
        source, source_rows = _source(tmp_path, name)
        sources.append(source)
        rows += source_rows
    partitions = tmp_path / "partitions.jsonl"
    partitions.write_text("".join(json.dumps(row) + "\n" for row in rows))
    inputs = tmp_path / "inputs.json"
    inputs.write_text(
        json.dumps(
            {
                "format": "speck_ladder_inputs",
                "format_version": 1,
                "partitions": {"path": str(partitions), "sha256": file_sha256(partitions)},
                "sources": sources,
            }
        )
    )
    return inputs


def _source_manifest(packed, source_id):
    manifest = json.loads((packed / "manifest.json").read_text())
    return next(source for source in manifest["sources"] if source["id"] == source_id)


def _train_passes(packed, source):
    """Split a source's train stream into per-pass lists of document token tuples."""
    shards = source["splits"]["train"]["shards"]
    stream = np.concatenate([np.fromfile(packed / shard["path"], dtype="<u2") for shard in shards])
    starts = np.cumsum([0] + [shard["tokens"] for shard in shards])
    orders = source["repetition"]["pass_orders"]
    bounds = [int(starts[entry["first_shard"]]) for entry in orders] + [len(stream)]
    passes = []
    for first, last in zip(bounds, bounds[1:]):
        tokens = stream[first:last].tolist()
        documents, current = [], []
        for token in tokens:
            current.append(token)
            if token == 2:
                documents.append(tuple(current))
                current = []
        assert not current
        passes.append(documents)
    return passes


def test_declared_passes_repeat_a_scarce_pool_in_distinct_seeded_orders(tmp_path, monkeypatch):
    monkeypatch.setattr(ladder, "get_tokenizer", lambda **_: Tokenizer())
    inputs = _inputs(tmp_path, "web", "code")
    weights = {"web": 20, "code": 80}
    # Without a declaration the scarce code bank fails rather than silently repeating.
    with pytest.raises(RuntimeError, match="code was exhausted"):
        _experiment(tmp_path, "undeclared", weights, inputs, tokens=1000)

    _, packed = _experiment(tmp_path, "repeated", weights, inputs, tokens=1000, passes={"code": 4})
    code = _source_manifest(packed, "code")
    train = code["splits"]["train"]
    repetition = code["repetition"]
    assert repetition["passes"] == 4
    assert repetition["unique_target_tokens"] == 200
    assert (
        repetition["unique_documents"]
        == train["documents"]
        == len([r for r in _records(packed)["code"] if r["split"] == "train"])
    )
    assert repetition["unique_tokens"] >= 200
    assert repetition["exposure_tokens"] == train["tokens"] >= train["requested_tokens"] == 800
    assert train["preparation_target_tokens"] == 800
    orders = repetition["pass_orders"]
    assert [entry["pass"] for entry in orders] == [1, 2, 3, 4]
    assert sum(entry["tokens"] for entry in orders) == train["tokens"]
    assert orders[0]["tokens"] == repetition["unique_tokens"]
    assert len({entry["order_sha256"] for entry in orders}) == 4
    assert "repetition" not in _source_manifest(packed, "web")

    passes = _train_passes(packed, code)
    pool = passes[0]
    assert len(pool) == repetition["unique_documents"] > 2
    for index, documents in enumerate(passes[1:], start=2):
        assert documents != pool[: len(documents)]
        if index < 4:
            # Full passes are permutations of exactly the unique pool.
            assert sorted(documents) == sorted(pool)
        else:
            assert set(documents) <= set(pool)
    assert passes[1] != passes[2]

    # The same declaration packs byte-identical shards and manifest entries.
    _, again = _experiment(tmp_path, "again", weights, inputs, tokens=1000, passes={"code": 4})
    assert _source_manifest(again, "code") == code


def test_repetition_leaves_validation_documents_and_bytes_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(ladder, "get_tokenizer", lambda **_: Tokenizer())
    inputs = _inputs(tmp_path, "web", "code")
    _, plain = _experiment(tmp_path, "plain", {"web": 50, "code": 50}, inputs)
    _, repeated = _experiment(
        tmp_path, "repeated", {"web": 20, "code": 80}, inputs, tokens=1000, passes={"code": 4}
    )
    for source_id in ("web", "code"):
        before, after = _source_manifest(plain, source_id), _source_manifest(repeated, source_id)
        assert before["splits"]["val"] == after["splits"]["val"]
        validation = [
            [r["content_hash"] for r in _records(packed)[source_id] if r["split"] == "val"]
            for packed in (plain, repeated)
        ]
        assert validation[0] and validation[0] == validation[1]


def test_a_pool_smaller_than_declared_fails_explicitly(tmp_path, monkeypatch):
    monkeypatch.setattr(ladder, "get_tokenizer", lambda **_: Tokenizer())
    inputs = _inputs(tmp_path, "web", "code")
    # Two passes need a 400-token unique code pool; its train families hold fewer tokens.
    with pytest.raises(RuntimeError, match="unique pool for 2 passes"):
        _experiment(
            tmp_path, "short", {"web": 20, "code": 80}, inputs, tokens=1000, passes={"code": 2}
        )
