import hashlib
import json

import numpy as np
import pytest
import torch

import speck.search.segments as segments_module
from speck.dataloader import manifest_fingerprint
from speck.search.segments import (
    DocumentRecord,
    SegmentPartition,
    SegmentPlan,
    TokenSpan,
    build_segment_plan,
    load_document_index,
    load_segment_plan,
    segment_evaluation_batches,
    segment_loader,
    validate_segment_plan,
)


def record(name, split, start, end):
    return DocumentRecord(
        hashlib.sha256(name.encode()).hexdigest(),
        split,
        start,
        end,
        "test",
        1.0,
    )


def records():
    return tuple(
        [record(f"train-{index}", "train", index * 10, index * 10 + 10) for index in range(8)]
        + [record(f"val-{index}", "val", index * 10, index * 10 + 10) for index in range(8)]
    )


def test_segment_plans_are_deterministic_and_disjoint():
    first = build_segment_plan(
        records(),
        "dataset",
        42,
        30,
        {"monitor": 10, "promotion": 20, "audit": 10, "final": 10},
    )
    second = build_segment_plan(
        records(),
        "dataset",
        42,
        30,
        {"monitor": 10, "promotion": 20, "audit": 10, "final": 10},
    )
    assert first == second
    assert first.digest == second.digest
    assert SegmentPlan.from_dict(first.export()) == first
    validation_hashes = [
        span.content_hash
        for partition in first.partitions
        if partition.split == "val"
        for span in partition.spans
    ]
    assert len(validation_hashes) == len(set(validation_hashes))


def test_segment_plan_changes_with_the_data_seed():
    first = build_segment_plan(records(), "dataset", 42, 30, {"audit": 20})
    second = build_segment_plan(records(), "dataset", 43, 30, {"audit": 20})
    assert first.digest != second.digest


def test_segment_plan_rejects_insufficient_documents():
    with pytest.raises(ValueError, match="cannot satisfy"):
        build_segment_plan(records(), "dataset", 42, 100, {"audit": 10})


def test_document_index_detects_tampering(tmp_path):
    path = tmp_path / "documents.jsonl"
    line = '{"content_hash":"' + "0" * 64 + '","end_token":2,"score":1.0,"source":"test","split":"train","start_token":0}\n'
    path.write_text(line)
    manifest = {
        "document_index": {
            "path": path.name,
            "records": 1,
            "sha256": hashlib.sha256(line.encode()).hexdigest(),
        }
    }
    assert len(load_document_index(tmp_path, manifest)) == 1
    path.write_text(line.replace('"end_token":2', '"end_token":3'))
    with pytest.raises(ValueError, match="checksum"):
        load_document_index(tmp_path, manifest)


def test_segment_plan_loads_and_matches_document_index(tmp_path):
    plan = build_segment_plan(records(), "dataset", 42, 20, {"monitor": 10})
    path = tmp_path / "segments.json"
    path.write_text(json.dumps(plan.export()))
    loaded = load_segment_plan(path)
    assert loaded == plan
    assert validate_segment_plan(loaded, records(), ("train", "monitor"))
    with pytest.raises(ValueError, match="missing partitions"):
        validate_segment_plan(loaded, records(), ("audit",))
    with pytest.raises(ValueError, match="does not match"):
        validate_segment_plan(loaded, records()[:1], ("train", "monitor"))


class FakeTokenizer:
    vocab_size = 32

    def fingerprint(self):
        return "1" * 64


def packed_segment_data(tmp_path):
    values = np.arange(1, 25, dtype="<u2")
    shards = []
    for index, shard_values in enumerate((values[:11], values[11:])):
        path = tmp_path / f"train_{index:05d}.bin"
        path.write_bytes(shard_values.tobytes())
        shards.append(
            {
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "tokens": len(shard_values),
            }
        )
    indexed = [
        record(f"document-{index}", "train", index * 6, index * 6 + 6)
        for index in range(4)
    ]
    lines = "".join(
        json.dumps(
            {
                "content_hash": item.content_hash,
                "end_token": item.end_token,
                "score": item.score,
                "source": item.source,
                "split": item.split,
                "start_token": item.start_token,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for item in indexed
    )
    (tmp_path / "documents.jsonl").write_text(lines)
    manifest = {
        "document_index": {
            "path": "documents.jsonl",
            "records": len(indexed),
            "sha256": hashlib.sha256(lines.encode()).hexdigest(),
        },
        "format": "speck_packed_tokens",
        "format_version": 2,
        "splits": {
            "train": {"shards": shards, "tokens": len(values)},
            "val": {"shards": shards, "tokens": len(values)},
        },
        "tokenizer": {"fingerprint": "1" * 64, "vocab_size": 32},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    plan = SegmentPlan(
        manifest_fingerprint(manifest),
        42,
        (
            SegmentPartition(
                "train",
                "train",
                tuple(
                    TokenSpan(item.content_hash, item.start_token, item.end_token)
                    for item in indexed
                ),
            ),
        ),
    )
    return plan


def test_segment_loader_replays_its_exact_cursor(tmp_path, monkeypatch):
    plan = packed_segment_data(tmp_path)
    calls = 0
    original = segments_module._ordered_spans

    def ordered_spans(*args):
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr(segments_module, "_ordered_spans", ordered_spans)
    loader = segment_loader(
        FakeTokenizer(), plan, "train", 7, 1, 3, device="cpu", data_dir=tmp_path
    )
    first = next(loader)
    second = next(loader)
    assert calls == 1
    resumed = segment_loader(
        FakeTokenizer(),
        plan,
        "train",
        7,
        1,
        3,
        device="cpu",
        data_dir=tmp_path,
        resume_state_dict=second[2],
    )
    replayed = next(resumed)
    assert torch.equal(second[0], replayed[0])
    assert torch.equal(second[1], replayed[1])
    assert second[2] == replayed[2]
    alternate = next(
        segment_loader(
            FakeTokenizer(), plan, "train", 8, 1, 3, device="cpu", data_dir=tmp_path
        )
    )
    assert first[2]["permutation"] != alternate[2]["permutation"]
    with pytest.raises(ValueError, match="changed segment state"):
        next(
            segment_loader(
                FakeTokenizer(),
                plan,
                "train",
                8,
                1,
                3,
                device="cpu",
                data_dir=tmp_path,
                resume_state_dict=second[2],
            )
        )


def test_segment_evaluation_covers_every_next_token_once(tmp_path):
    plan = packed_segment_data(tmp_path)
    batches = tuple(
        segment_evaluation_batches(
            FakeTokenizer(),
            plan,
            "train",
            2,
            4,
            device="cpu",
            data_dir=tmp_path,
        )
    )
    assert sum(labels.numel() for _, labels in batches) == 23
    assert all(
        torch.equal(inputs[:, 1:], labels[:, :-1])
        for inputs, labels in batches
    )
