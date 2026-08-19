import json

import numpy as np
import torch

import speck.dataset as dataset
import speck.dataloader as dataloader
from speck.dataloader import packed_loader


class FakeTokenizer:
    vocab_size = 32000
    bos_id = 1
    eos_id = 2

    def encode(self, text, bos=False, eos=False):
        tokens = [byte + 3 for byte in text.encode()]
        return ([1] if bos else []) + tokens + ([2] if eos else [])

    def fingerprint(self):
        return "test-tokenizer"


def make_tokenizer():
    return FakeTokenizer()


def test_prepare_and_resume_packed_dataset(tmp_path, monkeypatch):
    tokenizer = make_tokenizer()
    monkeypatch.setattr(dataset, "get_tokenizer", lambda: tokenizer)
    monkeypatch.setattr(
        dataset,
        "_is_validation_document",
        lambda content, seed, fraction: content.startswith("validation"),
    )
    documents = [
        {"content": "validation document " * 10, "score": 0.9, "source": "test"},
        {"content": "training document alpha beta gamma " * 20, "score": 0.9, "source": "test"},
    ] * 20
    manifest = dataset.prepare_dataset(
        train_tokens=200,
        validation_tokens=80,
        shard_tokens=37,
        output_dir=tmp_path,
        document_iterator=iter(documents),
    )
    assert manifest["splits"]["train"]["tokens"] >= 200
    assert manifest["splits"]["val"]["tokens"] >= 80
    for split in ("train", "val"):
        for shard in manifest["splits"][split]["shards"]:
            values = np.memmap(tmp_path / shard["path"], mode="r", dtype="<u2")
            assert len(values) == shard["tokens"]

    loader = packed_loader(
        tokenizer, 2, 8, "train", device="cpu", data_dir=tmp_path
    )
    inputs, targets, state = next(loader)
    expected_inputs, expected_targets, next_state = next(loader)
    assert inputs.shape == targets.shape == (2, 8)
    assert inputs.dtype == targets.dtype == torch.int64
    assert torch.equal(inputs[:, 1:], targets[:, :-1])

    resumed = packed_loader(
        tokenizer,
        2,
        8,
        "train",
        device="cpu",
        data_dir=tmp_path,
        resume_state_dict=next_state,
    )
    resumed_inputs, resumed_targets, resumed_state = next(resumed)
    assert torch.equal(resumed_inputs, expected_inputs)
    assert torch.equal(resumed_targets, expected_targets)
    assert resumed_state == next_state


def test_manifest_rejects_a_different_tokenizer(tmp_path, monkeypatch):
    tokenizer = make_tokenizer()
    monkeypatch.setattr(dataset, "get_tokenizer", lambda: tokenizer)
    monkeypatch.setattr(dataset, "_is_validation_document", lambda *args: "val" in args[0])
    documents = iter([
        {"content": "val " * 100, "score": 1.0, "source": "test"},
        {"content": "train " * 100, "score": 1.0, "source": "test"},
    ])
    dataset.prepare_dataset(
        train_tokens=10,
        validation_tokens=10,
        shard_tokens=8,
        output_dir=tmp_path,
        document_iterator=documents,
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["tokenizer"]["fingerprint"] = "wrong"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    loader = packed_loader(
        tokenizer, 1, 4, "train", device="cpu", data_dir=tmp_path
    )
    try:
        next(loader)
    except ValueError as error:
        assert "different tokenizer" in str(error)
    else:
        raise AssertionError("tokenizer mismatch was accepted")


def test_distributed_ranks_wrap_together(tmp_path, monkeypatch):
    tokenizer = make_tokenizer()
    monkeypatch.setattr(dataset, "get_tokenizer", lambda: tokenizer)
    monkeypatch.setattr(dataset, "_is_validation_document", lambda *args: "val" in args[0])
    documents = ([
        {"content": "val alpha beta " * 20, "score": 1.0, "source": "test"},
        {"content": "train alpha beta gamma " * 40, "score": 1.0, "source": "test"},
    ] * 20)
    dataset.prepare_dataset(
        train_tokens=90,
        validation_tokens=20,
        shard_tokens=31,
        output_dir=tmp_path,
        document_iterator=iter(documents),
    )

    monkeypatch.setattr(dataloader, "dist_info", lambda: (0, 0, 2))
    rank0 = packed_loader(
        tokenizer, 1, 8, "train", device="cpu", data_dir=tmp_path
    )
    _, _, state0 = next(rank0)
    monkeypatch.setattr(dataloader, "dist_info", lambda: (1, 1, 2))
    rank1 = packed_loader(
        tokenizer, 1, 8, "train", device="cpu", data_dir=tmp_path
    )
    _, _, state1 = next(rank1)
    assert state0 == state1
    for _ in range(10):
        _, _, state0 = next(rank0)
        _, _, state1 = next(rank1)
        assert state0 == state1


def test_shard_checksum_detects_corruption(tmp_path, monkeypatch):
    tokenizer = make_tokenizer()
    monkeypatch.setattr(dataset, "get_tokenizer", lambda: tokenizer)
    monkeypatch.setattr(dataset, "_is_validation_document", lambda *args: "val" in args[0])
    dataset.prepare_dataset(
        train_tokens=10,
        validation_tokens=10,
        shard_tokens=8,
        output_dir=tmp_path,
        document_iterator=iter([
            {"content": "val " * 100, "score": 1.0, "source": "test"},
            {"content": "train " * 100, "score": 1.0, "source": "test"},
        ]),
    )
    manifest = dataset.load_manifest(tmp_path)
    path = tmp_path / manifest["splits"]["train"]["shards"][0]["path"]
    with path.open("r+b") as handle:
        first = handle.read(1)
        handle.seek(0)
        handle.write(bytes([first[0] ^ 1]))
    try:
        dataset.verify_shards(tmp_path, manifest)
    except ValueError as error:
        assert "checksum mismatch" in str(error)
    else:
        raise AssertionError("corrupted shard was accepted")
