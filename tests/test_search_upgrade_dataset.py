import hashlib
import json

import numpy as np
import pytest

from speck.search.segments import load_document_index
from speck.search.upgrade_dataset import upgrade_document_index


def write_shard(path, values):
    values = np.asarray(values, dtype="<u2")
    path.write_bytes(values.tobytes())
    return {
        "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "tokens": len(values),
    }


def format_one_dataset(tmp_path, malformed=False):
    train = [1, 5, 2, 1, 6, 7, 2]
    if malformed:
        train = [1, 5, 1, 6, 2]
    train_shards = [
        write_shard(tmp_path / "train_00000.bin", train[:5]),
        write_shard(tmp_path / "train_00001.bin", train[5:]),
    ]
    val = [1, 8, 2]
    manifest = {
        "dataset": {"repo": "test"},
        "documents": 3,
        "dtype": "<u2",
        "format": "speck_packed_tokens",
        "format_version": 1,
        "sources": {"test": 3},
        "splits": {
            "train": {"shards": train_shards, "tokens": len(train)},
            "val": {
                "shards": [write_shard(tmp_path / "val_00000.bin", val)],
                "tokens": len(val),
            },
        },
        "tokenizer": {
            "bos_token_id": 1,
            "eos_token_id": 2,
            "fingerprint": "tokenizer",
            "vocab_size": 16,
        },
    }
    encoded = json.dumps(manifest, indent=2, sort_keys=True).encode()
    (tmp_path / "manifest.json").write_bytes(encoded)
    return encoded


def test_format_one_upgrade_recovers_cross_shard_documents(tmp_path):
    original = format_one_dataset(tmp_path)
    result = upgrade_document_index(tmp_path)
    assert result["upgraded"]
    assert (tmp_path / "manifest.v1.json").read_bytes() == original
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["format_version"] == 2
    records = load_document_index(tmp_path, manifest)
    assert [(value.split, value.start_token, value.end_token) for value in records] == [
        ("train", 0, 3),
        ("train", 3, 7),
        ("val", 0, 3),
    ]
    assert not upgrade_document_index(tmp_path)["upgraded"]


def test_format_one_upgrade_rejects_malformed_boundaries(tmp_path):
    format_one_dataset(tmp_path, malformed=True)
    with pytest.raises(ValueError, match="bos and eos counts"):
        upgrade_document_index(tmp_path)
    assert json.loads((tmp_path / "manifest.json").read_text())["format_version"] == 1
    assert not (tmp_path / "documents.jsonl").exists()
