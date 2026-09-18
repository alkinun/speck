import os
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from speck.evaluation.capability import selected_rows
from speck.evaluation.code_runner import check_sandbox, run_python
from speck.provenance.io import file_sha256


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
