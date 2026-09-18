import json
import os
import shutil
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from speck.evaluation.capability import model_settings, selected_rows, validate_eos_boundary
from speck.evaluation.code_runner import check_sandbox, run_python
from speck.provenance.io import file_sha256


def test_local_capability_model_requires_parity_and_binds_payload(tmp_path):
    args = SimpleNamespace(local_export=tmp_path)
    with pytest.raises(ValueError, match="missing parity artifacts"):
        model_settings(args)
    (tmp_path / "config.json").write_text(json.dumps({"expected_parameters": 7}))
    (tmp_path / "model.safetensors").write_bytes(b"test weights")
    parity = {"format": "speck_export_parity", "passed": False, "parameters": 7}
    (tmp_path / "speck_parity.json").write_text(json.dumps(parity))
    with pytest.raises(ValueError, match="no successful"):
        model_settings(args)
    parity["passed"] = True
    (tmp_path / "speck_parity.json").write_text(json.dumps(parity))
    backend, identity = model_settings(args)
    assert backend == {
        "pretrained": str(tmp_path.resolve()),
        "trust_remote_code": True,
        "local_files_only": True,
        "use_cache": False,
    }
    (tmp_path / "model.safetensors").write_bytes(b"different weights")
    assert model_settings(args)[1]["sha256"] != identity["sha256"]
    parity["parameters"] = 8
    (tmp_path / "speck_parity.json").write_text(json.dumps(parity))
    with pytest.raises(ValueError, match="does not match"):
        model_settings(args)


def test_hub_capability_model_stays_commit_pinned_without_custom_code():
    args = SimpleNamespace(local_export=None, model="reference/model", revision="a" * 40)
    backend, identity = model_settings(args)
    assert backend["revision"] == args.revision
    assert backend["trust_remote_code"] is False and identity is None
    args.revision = "main"
    with pytest.raises(ValueError, match="full commit hash"):
        model_settings(args)


def test_evaluator_rejects_an_empty_eos_stop_string():
    model = SimpleNamespace(eot_token_id=2, tok_decode=lambda *args, **kwargs: "")
    with pytest.raises(ValueError, match="EOS to empty"):
        validate_eos_boundary(model)
    model.tok_decode = lambda *args, **kwargs: "</s>"
    validate_eos_boundary(model)


def test_frozen_rows_keep_grading_fields_and_reject_changed_bytes(tmp_path):
    path = tmp_path / "tasks.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"id": "dev", "question": "2+2?", "answerKey": "B"},
                {"id": "final", "question": "3+3?", "answerKey": "C"},
            ]
        ),
        path,
    )
    benchmark = {
        "id": "arc",
        "format": "parquet",
        "path": str(path),
        "sha256": file_sha256(path),
        "task_id_field": "id",
        "expected_tasks": 2,
    }
    prepared = {"partitions": {"arc": {"development": ["dev"], "final": ["final"]}}}
    rows = selected_rows(prepared, benchmark, "development", 0)
    assert len(rows) == 1 and rows[0]["answerKey"] == "B"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes changed"):
        selected_rows(prepared, benchmark, "development", 0)


def test_code_execution_refuses_missing_isolation(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="refusing unsandboxed"):
        run_python("raise Exception()", "", "f")


def test_code_execution_requires_enforceable_process_limits(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/bwrap")
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    with pytest.raises(RuntimeError, match="non-root"):
        check_sandbox()
