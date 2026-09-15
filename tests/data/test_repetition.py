import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from speck.data import loader, repetition
from speck.data.loader import PackedTokenSource, loader_state_for_offset, packed_loader
from speck.data.packing import TokenShardWriter
from speck.provenance.io import durable_json, file_sha256


def identity(path):
    return {"path": str(path), "sha256": file_sha256(path)}


@pytest.fixture
def prepared(tmp_path):
    stock = tmp_path / "stock"
    writer = TokenShardWriter(stock, "tokens", 23)
    rows, expected = [], []
    for i in range(12):
        values = [1, *range(10 + i * 14, 24 + i * 14), 2]
        rows.append(
            {
                "ordinal": i,
                "content_id": str(i),
                "released_content_sha256": hashlib.sha256(str(i).encode()).hexdigest(),
                "token_start": len(expected),
                "token_count": len(values),
                "utf8_bytes": 14,
            }
        )
        expected.extend(values)
        writer.write(values)
    shards = writer.finish()
    documents = stock / "documents.jsonl"
    documents.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {
        "format": "speck_document_token_stock",
        "format_version": 1,
        "status": "complete_document_token_cache_not_training_view",
        "plan": {
            "source_id": "fixture",
            "expected_documents": 12,
            "expected_tokens": 192,
            "tokenizer": {"path": "fixture.model", "sha256": "fixture-tokenizer"},
        },
        "document_count": 12,
        "token_count": 192,
        "documents": {"path": documents.name, "sha256": file_sha256(documents)},
        "shards": shards,
        "training_authority": False,
    }
    durable_json(stock / "manifest.json", manifest)
    # Explicitly reorder stock documents, so tests catch accidentally reading source prefixes.
    order = [5, 0, 8, 2, 10, 3, 11, 4, 9, 6, 1, 7]
    order_path = tmp_path / "order.jsonl"
    order_path.write_text("".join(json.dumps(i) + "\n" for i in order))
    plan = {
        "format": "speck_repetition_source_plan",
        "format_version": 1,
        "stock_manifest": identity(stock / "manifest.json"),
        "document_order": identity(order_path),
        "source_id": "fixture",
        "seed": 42,
        "exposure_tokens": 192,
        "global_stride": 8,
        "effective_epochs": [1, 2, 4],
        "shard_tokens": 17,
        "output_directory": str(tmp_path / "result"),
        "training_authority": False,
    }
    plan_path = tmp_path / "plan.json"
    durable_json(plan_path, plan)
    ordered = np.concatenate(
        [np.array(expected[i * 16 : (i + 1) * 16], dtype="<u2") for i in order]
    )
    return plan_path, plan, ordered


def source(output, variant):
    return PackedTokenSource(output, {"id": "fixture", "splits": {"train": variant}}, "train")


def test_exact_nested_whole_documents_exposures_and_lookahead(prepared):
    path, plan, ordered = prepared
    report = repetition.materialize_repetition(path)
    assert report["training_authority"] is False
    for epochs in (1, 2, 4):
        variant = report["variants"][str(epochs)]
        length = 192 // epochs
        expected = np.concatenate([np.tile(ordered[:length], epochs), ordered[:1]])
        np.testing.assert_array_equal(
            source(plan["output_directory"], variant).read(0, 193), expected
        )
        assert variant["unique_pool_documents"] == 12 // epochs
        assert variant["unique_pool_tokens"] * epochs == variant["input_exposure_tokens"] == 192
        # Both inputs and shifted targets cover each pool token position exactly e times.
        assert sorted(expected[:-1]) == sorted(expected[1:])
    assert report["variants"]["4"]["lookahead"]["additional_training_inputs"] == 0


def test_interrupted_shards_and_partial_attempts_preserved_with_clean_replay(prepared, tmp_path):
    path, plan, _ = prepared
    with pytest.raises(RuntimeError, match="injected"):
        repetition.materialize_repetition(path, interrupt_after_shards=2)
    output = Path(plan["output_directory"])
    complete = output / "epochs-1/shard-00000/manifest.json"
    original = complete.read_bytes()
    pending = output / "epochs-1/shard-00002"
    pending.mkdir()
    tail = pending / "attempt-00000.bin"
    tail.write_bytes(b"power-loss-tail")
    report = repetition.materialize_repetition(path, resume=True)
    assert tail.read_bytes() == b"power-loss-tail"
    assert (pending / "attempt-00001.bin").exists()
    assert complete.read_bytes() == original
    assert repetition.materialize_repetition(path, resume=True) == report
    clean_plan = {**plan, "output_directory": str(tmp_path / "clean")}
    clean_path = tmp_path / "clean.json"
    durable_json(clean_path, clean_plan)
    clean = repetition.materialize_repetition(clean_path)
    for epoch in ("1", "2", "4"):
        assert [s["sha256"] for s in clean["variants"][epoch]["shards"]] == [
            s["sha256"] for s in report["variants"][epoch]["shards"]
        ]


@pytest.mark.parametrize(
    "change", ["alignment", "boundary", "duplicate", "absent", "source", "changed_stock"]
)
def test_invalid_pools_rejected_before_any_output(prepared, change):
    path, plan, _ = prepared
    if change == "alignment":
        plan["global_stride"] = 5
    elif change == "boundary":
        # Total 192 is divisible by 4 * 8, but move two lengths so quarter ends mid-document.
        stock_path = Path(plan["stock_manifest"]["path"])
        stock = json.loads(stock_path.read_text())
        index = stock_path.parent / stock["documents"]["path"]
        rows = [json.loads(line) for line in index.read_text().splitlines()]
        rows[0]["token_count"] -= 1
        rows[1]["token_count"] += 1
        rows[1]["token_start"] -= 1
        index.write_text("".join(json.dumps(r) + "\n" for r in rows))
        stock["documents"]["sha256"] = file_sha256(index)
        durable_json(stock_path, stock)
        plan["stock_manifest"] = identity(stock_path)
    elif change in ("duplicate", "absent"):
        index = Path(plan["document_order"]["path"])
        index.write_text("0\n0\n" if change == "duplicate" else "99\n")
        plan["document_order"] = identity(index)
    elif change == "source":
        plan["source_id"] = "other"
    else:
        stock_path = Path(plan["stock_manifest"]["path"])
        next(stock_path.parent.glob("tokens_*.bin")).write_bytes(b"bad")
    durable_json(path, plan)
    with pytest.raises(ValueError):
        repetition.materialize_repetition(path)
    assert not Path(plan["output_directory"]).exists()


def test_resume_rejects_changed_plan_or_output(prepared):
    path, plan, _ = prepared
    report = repetition.materialize_repetition(path)
    plan["seed"] = 43
    durable_json(path, plan)
    with pytest.raises(ValueError, match="identical repetition plan"):
        repetition.materialize_repetition(path, resume=True)
    plan["seed"] = 42
    durable_json(path, plan)
    shard = Path(plan["output_directory"]) / report["variants"]["2"]["shards"][0]["path"]
    shard.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="repetition shard"):
        repetition.materialize_repetition(path, resume=True)


def test_loader_algorithm_reads_every_rank_and_resumes_across_repetition_boundaries(
    prepared, monkeypatch
):
    path, plan, ordered = prepared
    report = repetition.materialize_repetition(path)
    tokenizer = SimpleNamespace(vocab_size=256, fingerprint=lambda: "fixture-tokenizer")
    # This fixture isolates the unchanged loader algorithm. Full packed-manifest export,
    # joint eligibility and launch receipts are intentionally not supplied by this component.
    for epochs in (1, 2, 4):
        variant = report["variants"][str(epochs)]
        manifest = {
            "requested_train_tokens": 192,
            "tokenizer": {"vocab_size": 256, "fingerprint": "fixture-tokenizer"},
            "mixture": {"phases": [{"end_tokens": 192, "weights": {"fixture": 1}}]},
            "sources": [{"id": "fixture", "splits": {"train": variant}}],
        }
        monkeypatch.setattr(loader, "load_manifest", lambda *a, m=manifest: m)
        expected = np.concatenate([np.tile(ordered[: 192 // epochs], epochs), ordered[:1]])
        all_inputs = []
        running = {}
        for offset in range(0, 192, 8):
            batch_inputs = []
            for rank in (0, 1):
                monkeypatch.setattr(loader, "dist_info", lambda r=rank: (r, r, 2))
                state = loader_state_for_offset(manifest, "train", offset, 2, 2, 2)
                generator = packed_loader(
                    tokenizer,
                    2,
                    2,
                    device="cpu",
                    data_dir=plan["output_directory"],
                    resume_state_dict=state,
                )
                inputs, targets, actual = next(generator)
                if rank not in running:
                    running[rank] = packed_loader(
                        tokenizer, 2, 2, device="cpu", data_dir=plan["output_directory"]
                    )
                continuous_inputs, continuous_targets, continuous_state = next(running[rank])
                np.testing.assert_array_equal(inputs, continuous_inputs)
                np.testing.assert_array_equal(targets, continuous_targets)
                assert actual == continuous_state
                np.testing.assert_array_equal(
                    inputs.flatten(), expected[offset + rank * 4 : offset + rank * 4 + 4]
                )
                np.testing.assert_array_equal(
                    targets.flatten(), expected[offset + rank * 4 + 1 : offset + rank * 4 + 5]
                )
                assert actual == state
                assert actual["source_epochs"] == {"fixture": 0}
                batch_inputs.extend(inputs.flatten().tolist())
            all_inputs.extend(batch_inputs)
        np.testing.assert_array_equal(all_inputs, expected[:-1])
        # The missing final lookahead remains a capacity error in the unchanged loader.
        broken = json.loads(json.dumps(manifest))
        broken["sources"][0]["splits"]["train"]["tokens"] -= 1
        with pytest.raises(ValueError, match="too small"):
            loader._validate_training_capacity(broken, 8)
