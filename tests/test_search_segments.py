import hashlib
import json

import pytest

from speck.search.segments import (
    DocumentRecord,
    SegmentPlan,
    build_segment_plan,
    load_document_index,
    load_segment_plan,
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
