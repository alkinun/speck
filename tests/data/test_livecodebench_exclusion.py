"""Exclusion projection must preserve coverage and never admit private test payloads."""

import hashlib
import json

import pytest

from speck.data.sources.livecodebench_exclusion import project_files, project_row


def row(question="1", platform="example"):
    return {
        "platform": platform,
        "question_id": question,
        "contest_date": "2025-01-01T12:00:00",
        "question_title": "Test task",
        "question_content": "Synthetic test problem, not a benchmark.",
        "starter_code": "",
        "public_test_cases": "[]",
        "private_test_cases": "DO NOT DECODE",
    }


def artifact(tmp_path, name, rows):
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_complete_projection_and_platform_scoped_identity(tmp_path):
    source = artifact(tmp_path, "raw.jsonl", [row(), row(platform="other")])
    target = tmp_path / "projection.jsonl"
    receipt = project_files([source], target)
    projected = [json.loads(line) for line in target.read_text().splitlines()]
    assert receipt["rows"] == receipt["inputs"][0]["rows"] == 2
    assert len({r["task_id"] for r in projected}) == 2
    assert "DO NOT DECODE" not in target.read_text()
    assert all("private_test_cases" not in r for r in projected)
    assert all(r["source_row"] == i for i, r in enumerate(projected))
    assert receipt["training_admitted"] is False


def test_duplicate_across_files_fails_without_output(tmp_path):
    files = [artifact(tmp_path, name, [row()]) for name in ["a.jsonl", "b.jsonl"]]
    target = tmp_path / "projection.jsonl"
    with pytest.raises(ValueError, match="duplicate"):
        project_files(files, target)
    assert not target.exists() and not target.with_suffix(".jsonl.building").exists()


def test_bad_hash_is_rejected_before_output(tmp_path):
    source = artifact(tmp_path, "raw.jsonl", [row()])
    source["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity mismatch"):
        project_files([source], tmp_path / "projection.jsonl")


@pytest.mark.parametrize(
    "changes",
    [
        {"question_id": ""},
        {"question_content": " "},
        {"contest_date": "not-a-date"},
        {"public_test_cases": []},
        {"platform": None},
    ],
)
def test_bad_schema_or_identity_is_rejected(changes):
    with pytest.raises(ValueError):
        project_row({**row(), **changes})


def test_existing_output_is_preserved(tmp_path):
    output = tmp_path / "projection.jsonl"
    output.write_text("existing")
    with pytest.raises(FileExistsError):
        project_files([artifact(tmp_path, "raw.jsonl", [row()])], output)
    assert output.read_text() == "existing"
