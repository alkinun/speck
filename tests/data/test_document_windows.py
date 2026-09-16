"""Document isolation, finite exposure and exact distributed resume checks."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from speck.data.dataset import TokenShardWriter
from speck.data.document_windows import DocumentWindowReader, prepare_document_windows
from speck.provenance.io import durable_json, file_sha256


def identity(path):
    return {"path": str(path), "sha256": file_sha256(path)}


@pytest.fixture
def inputs(tmp_path):
    stock = tmp_path / "stock"
    writer = TokenShardWriter(stock, "tokens", 7)
    documents = []
    for ordinal in range(4):
        writer.write(list(range(ordinal * 20, ordinal * 20 + 10)))
        documents.append(
            {
                "ordinal": ordinal,
                "token_start": ordinal * 10,
                "token_count": 10,
                "utf8_bytes": 30,
                "content_id": None,
                "released_content_sha256": hashlib.sha256(str(ordinal).encode()).hexdigest(),
            }
        )
    index = stock / "documents.jsonl"
    index.write_text("".join(json.dumps(r) + "\n" for r in documents))
    manifest = {
        "format": "speck_document_token_stock",
        "format_version": 1,
        "status": "complete_document_token_cache_not_training_view",
        "training_authority": False,
        "document_count": 4,
        "token_count": 40,
        "documents": {"path": index.name, "sha256": file_sha256(index)},
        "shards": writer.finish(),
        "plan": {
            "source_id": "fixture",
            "category": "reference",
            "expected_documents": 4,
            "expected_tokens": 40,
            "tokenizer": {"sha256": "fixture"},
        },
    }
    durable_json(stock / "manifest.json", manifest)
    order = tmp_path / "windows.jsonl"
    order.write_text(
        "".join(json.dumps({"ordinal": i, "document_offset": 2}) + "\n" for i in (2, 0, 3, 1))
    )
    spec = {
        "format": "speck_document_window_preparation",
        "format_version": 1,
        "stock_manifest": identity(stock / "manifest.json"),
        "windows": identity(order),
        "sequence_length": 4,
        "output_directory": str(tmp_path / "view"),
        "training_authority": False,
    }
    path = tmp_path / "plan.json"
    durable_json(path, spec)
    return path, spec


def test_cross_shard_whole_document_windows_rank_disjoint_and_fresh_resume(inputs):
    path, spec = inputs
    report = prepare_document_windows(path)
    assert report["input_tokens"] == 16
    assert not report["family_partition_qualified"] and not report["model_isolation_qualified"]
    view = identity(Path(spec["output_directory"]) / "manifest.json")
    readers = [DocumentWindowReader(view, batch_size=1, world_size=2, rank=i) for i in (0, 1)]
    for batch, starts in enumerate(((42, 2), (62, 22))):
        for reader, start in zip(readers, starts, strict=True):
            x, y, rows, next_state = reader.read(reader.state(batch))
            np.testing.assert_array_equal(x.numpy(), [list(range(start, start + 4))])
            np.testing.assert_array_equal(y.numpy(), [list(range(start + 1, start + 5))])
            assert rows[0]["content_id"] is None
            assert next_state == reader.state(batch + 1)
    replay = DocumentWindowReader(view, batch_size=1, world_size=2, rank=0)
    original = readers[0].read(readers[0].state(1))
    restored = replay.read(json.loads(json.dumps(readers[0].state(1))))
    assert np.array_equal(original[0], restored[0]) and original[2:] == restored[2:]
    with pytest.raises(StopIteration):
        replay.read(replay.state(2))
    with pytest.raises(ValueError, match="geometry"):
        replay.read({**replay.state(1), "world_size": 1})
    with pytest.raises(ValueError, match="implicit drop"):
        DocumentWindowReader(view, batch_size=3)


@pytest.mark.parametrize(
    "rows",
    [
        [{"ordinal": 0, "document_offset": 6}],  # Four inputs fit; their lookahead does not.
        [{"ordinal": 0, "document_offset": 0}, {"ordinal": 0, "document_offset": 3}],
        [{"ordinal": 9, "document_offset": 0}],
        [{"ordinal": True, "document_offset": 0}],
    ],
)
def test_reject_boundary_crossing_overlap_absent_and_boolean_indices(inputs, rows):
    path, spec = inputs
    order = Path(spec["windows"]["path"])
    order.write_text("".join(json.dumps(r) + "\n" for r in rows))
    spec["windows"] = identity(order)
    durable_json(path, spec)
    with pytest.raises(ValueError):
        prepare_document_windows(path)
    assert not Path(spec["output_directory"]).exists()


def test_shared_lookahead_allowed_but_not_repeated_inputs(inputs):
    path, spec = inputs
    order = Path(spec["windows"]["path"])
    order.write_text(
        "{" + '"ordinal":0,"document_offset":0}\n' + '{"ordinal":0,"document_offset":4}\n'
    )
    spec["windows"] = identity(order)
    durable_json(path, spec)
    report = prepare_document_windows(path)
    assert [r["source_token_start"] for r in report["windows"]] == [0, 4]
    with pytest.raises(FileExistsError):
        prepare_document_windows(path)


def test_rebound_view_cannot_change_the_planned_document_span(inputs):
    path, spec = inputs
    report = prepare_document_windows(path)
    manifest = Path(spec["output_directory"]) / "manifest.json"
    report["windows"][0]["source_token_start"] += 8
    durable_json(manifest, report)
    with pytest.raises(ValueError, match="verified preparation"):
        DocumentWindowReader(identity(manifest), batch_size=1)


def test_same_size_changed_token_payload_rejected(inputs):
    path, spec = inputs
    prepare_document_windows(path)
    stock = Path(spec["stock_manifest"]["path"])
    shard = stock.parent / json.loads(stock.read_text())["shards"][0]["path"]
    shard.write_bytes(b"\xff" * shard.stat().st_size)
    with pytest.raises(ValueError, match="payload identity"):
        DocumentWindowReader(
            identity(Path(spec["output_directory"]) / "manifest.json"), batch_size=1
        )
