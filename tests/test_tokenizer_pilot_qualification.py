import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from speck import tokenizer_pilot_qualification as qualification


class FakeStream:
    def __init__(self, run, verify_hashes):
        assert verify_hashes is True
        self.run = run
        self.fixed_tokens = 12
        self.total_tokens = 25
        self.maximum_offset = 16
        self.shards = [object(), object()]
        self.shard_manifests = [
            {"path": run["tokenizer_id"] + "-a.bin", "sha256": "a"},
            {"path": run["tokenizer_id"] + "-b.bin", "sha256": "b"},
        ]

    def read(self, start, count, dtype=np.int64):
        return np.arange(start, start + count, dtype=dtype)


def identity(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def fake_run(run_id, shared):
    return {
        "run_id": run_id,
        "run_fingerprint": f"fingerprint-{run_id}",
        "tokenizer_id": run_id,
        "tokenizer": {"model": shared},
        "training_data": {"fixed_stream": shared, "continuation": shared},
        "evaluation": {
            "sample": shared,
            "categories": [
                {
                    "id": "web",
                    "path": shared["path"],
                    "sha256": shared["sha256"],
                    "documents": 1,
                    "utf8_bytes": 5,
                }
            ],
        },
        "flop_correction": shared,
        "settings": {
            "sequence_length": 2,
            "device_batch_size": 2,
            "batch_tokens": 8,
        },
        "stops": {"run_stop_aligned_tokens": 16},
    }


def test_qualification_checks_three_runs_evaluation_and_resume_boundaries(tmp_path, monkeypatch):
    shared_path = tmp_path / "shared.json"
    shared_path.write_text("shared\n")
    shared = identity(shared_path)
    runs = {f"run-{index}": fake_run(f"run-{index}", shared) for index in range(3)}
    materialization = {
        "format": "speck_tokenizer_pilot_corrected_screen_materialization_result",
        "format_version": 2,
        "status": "three_corrected_screen_runs_materialized_execution_blocked",
        "plan_fingerprint": "plan",
        "authority": {"screen_execution": False},
        "runs": [
            {
                "path": f"{run_id}.json",
                "run_id": run_id,
                "run_fingerprint": run["run_fingerprint"],
            }
            for run_id, run in runs.items()
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(materialization))
    monkeypatch.setattr(
        qualification,
        "load_pilot_run_manifest",
        lambda run_path: runs[Path(run_path).stem],
    )
    monkeypatch.setattr(qualification, "PilotTokenStream", FakeStream)
    monkeypatch.setattr(
        qualification,
        "iter_evaluation_documents",
        lambda category, verify_hash: iter(
            [{"document_id": "document", "text": "hello", "utf8_bytes": 5}]
        ),
    )

    result = qualification.qualify_corrected_runtime(path)

    assert result["status"] == (
        "corrected_streams_and_evaluation_inputs_qualified_no_model_outputs"
    )
    assert len(result["runs"]) == 3
    assert result["unique_evaluation_files_parsed"] == 1
    assert all(len(run["resume_probes"]) == 3 for run in result["runs"])
    assert result["gates"]["model_outputs_created"] == 0
    assert result["authority"]["screen_execution"] is False


def test_qualification_rejects_expanded_authority_or_run_identity(tmp_path):
    value = {
        "format": "speck_tokenizer_pilot_corrected_screen_materialization_result",
        "format_version": 2,
        "status": "three_corrected_screen_runs_materialized_execution_blocked",
        "authority": {"screen_execution": True},
        "runs": [],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="unsupported"):
        qualification.qualify_corrected_runtime(path)
