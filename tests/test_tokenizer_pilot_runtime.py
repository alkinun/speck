import hashlib

import numpy as np
import pytest
import torch

from speck.tokenizer_pilot_runs import _fingerprint
from speck.tokenizer_pilot_runtime import (
    PilotBatchLoader,
    PilotTokenStream,
    validate_pilot_run_manifest,
)


def shard(tmp_path, name, values, category):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(values, dtype="<u2")
    path.write_bytes(array.tobytes())
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "tokens": len(values),
        "category": category,
    }


def run_manifest(tmp_path):
    fixed = [
        shard(tmp_path, "fixed-a.bin", range(5), "web"),
        shard(tmp_path, "fixed-b.bin", range(5, 12), "code"),
    ]
    continuation = [shard(tmp_path, "continuation.bin", range(12, 25), "web")]
    value = {
        "format": "speck_tokenizer_pilot_corrected_screen_run",
        "format_version": 2,
        "status": "corrected_materialized_not_started_execution_blocked",
        "run_id": "fixture",
        "tokenizer_id": "fixture-tokenizer",
        "repository_revision": "fixture-revision",
        "training_data": {
            "fixed_stream": {"path": "fixed.json", "sha256": "fixed"},
            "continuation": {"path": "continuation.json", "sha256": "continuation"},
            "document_stream_sha256": "documents",
            "fixed_shards": fixed,
            "continuation_shards": continuation,
        },
        "settings": {
            "sequence_length": 2,
            "device_batch_size": 2,
            "batch_tokens": 8,
            "accumulation": 2,
        },
        "stops": {
            "fixed_document_tokens": 12,
            "fixed_document_step": 2,
            "fixed_flop_token_stop": 16,
            "run_stop_aligned_tokens": 16,
            "final_step": 2,
        },
        "authority": {
            "v2_run_materialization": True,
            "screen_execution": False,
            "confirmation_execution": False,
            "D5_opening": False,
            "final_selection": False,
            "flagship_training": False,
        },
    }
    value["run_fingerprint"] = _fingerprint(value)
    return value


def test_stream_reads_exactly_across_fixed_and_continuation_shards(tmp_path):
    stream = PilotTokenStream(run_manifest(tmp_path))

    assert stream.fixed_tokens == 12
    assert stream.total_tokens == 25
    assert stream.read(3, 15).tolist() == list(range(3, 18))
    assert stream.read(5, 0, dtype=np.uint16).dtype == np.uint16
    with pytest.raises(IndexError, match="out of range"):
        stream.read(24, 2)


def test_loader_yields_contiguous_batches_and_resumes_at_next_batch(tmp_path):
    stream = PilotTokenStream(run_manifest(tmp_path))
    loader = PilotBatchLoader(stream)

    first_inputs, first_targets, first_state = next(loader)
    second_state = loader.state_dict()
    second_inputs, second_targets, returned_state = next(loader)

    assert torch.equal(first_inputs, torch.tensor([[0, 1], [2, 3]]))
    assert torch.equal(first_targets, torch.tensor([[1, 2], [3, 4]]))
    assert first_state["token_offset"] == 0
    assert returned_state == second_state
    assert torch.equal(second_inputs, torch.tensor([[4, 5], [6, 7]]))
    assert torch.equal(second_targets, torch.tensor([[5, 6], [7, 8]]))

    resumed = PilotBatchLoader(stream, resume_state=second_state)
    resumed_inputs, resumed_targets, resumed_state = next(resumed)
    assert torch.equal(resumed_inputs, second_inputs)
    assert torch.equal(resumed_targets, second_targets)
    assert resumed_state == second_state

    final = PilotBatchLoader(stream, resume_state={**second_state, "token_offset": 16})
    with pytest.raises(StopIteration):
        next(final)


def test_resume_rejects_changed_run_geometry_or_unaligned_offset(tmp_path):
    stream = PilotTokenStream(run_manifest(tmp_path))
    state = PilotBatchLoader(stream).state_dict()
    state["token_offset"] = 1
    with pytest.raises(ValueError, match="offset"):
        PilotBatchLoader(stream, resume_state=state)

    state["token_offset"] = 0
    state["run_fingerprint"] = "other"
    with pytest.raises(ValueError, match="differs"):
        PilotBatchLoader(stream, resume_state=state)


def test_manifest_and_shard_mutations_fail_closed(tmp_path):
    run = run_manifest(tmp_path)
    run["stops"]["fixed_document_tokens"] += 1
    with pytest.raises(ValueError, match="fingerprint"):
        validate_pilot_run_manifest(run)

    run = run_manifest(tmp_path / "hash")
    path = run["training_data"]["fixed_shards"][0]["path"]
    with open(path, "r+b") as handle:
        handle.write(b"xx")
    with pytest.raises(ValueError, match="checksum"):
        PilotTokenStream(run)
