import json

import pytest

from speck.evaluation.deferred_code import validate_pending
from speck.provenance.io import atomic_json, file_sha256


@pytest.fixture
def pending_case(tmp_path):
    protocol = tmp_path / "protocol.json"
    atomic_json(protocol, {"protocol": "frozen"})
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps({"id": "dev", "prompt": "def f():", "test": "check", "entry_point": "f"}) + "\n"
    )
    prepared = {
        "protocol_sha256": file_sha256(protocol),
        "partitions": {"humanevalplus": {"development": ["dev"], "final": []}},
        "benchmarks": [
            {
                "id": "humanevalplus",
                "path": str(tasks),
                "sha256": file_sha256(tasks),
                "format": "jsonl",
                "expected_tasks": 1,
                "task_id_field": "id",
            }
        ],
    }
    local, source = tmp_path / "local.json", tmp_path / "source.json"
    atomic_json(local, prepared)
    prepared["benchmarks"][0]["path"] = "/remote/tasks.jsonl"
    atomic_json(source, prepared)
    pending = tmp_path / "pending"
    pending.mkdir()
    rows = [
        {
            "benchmark": "humanevalplus",
            "task_id": "dev",
            "metrics": {},
            "execution": {"status": "pending_local_grading"},
            "response": "\n return 1",
        }
    ]
    (pending / "outputs.jsonl").write_text(json.dumps(rows[0]) + "\n")
    result = {
        "status": "code_grading_pending",
        "defer_code_grading": True,
        "protocol_sha256": file_sha256(protocol),
        "prepared_sha256": file_sha256(source),
        "outputs_sha256": file_sha256(pending / "outputs.jsonl"),
        "partition": "development",
        "limit": 0,
    }
    atomic_json(pending / "result.json", result)
    return protocol, local, source, pending


def test_pending_grading_accepts_only_path_relocation(pending_case):
    result, docs, code = validate_pending(*pending_case)
    assert result["status"] == "code_grading_pending"
    assert docs[0]["speck_task_id"] == code[0]["task_id"] == "dev"


@pytest.mark.parametrize(
    "change", ["bytes", "duplicate", "missing", "pregraded", "partition", "completed"]
)
def test_pending_grading_rejects_changed_or_incomplete_inputs(pending_case, change):
    protocol, local, source, pending = pending_case
    outputs = pending / "outputs.jsonl"
    result = json.loads((pending / "result.json").read_text())
    if change == "bytes":
        outputs.write_text(outputs.read_text() + " ")
    elif change == "duplicate":
        outputs.write_text(outputs.read_text() * 2)
        result["outputs_sha256"] = file_sha256(outputs)
    elif change == "missing":
        outputs.write_text("")
        result["outputs_sha256"] = file_sha256(outputs)
    elif change == "pregraded":
        row = json.loads(outputs.read_text())
        row["metrics"] = {"compiled_plus_pass@1": True}
        outputs.write_text(json.dumps(row) + "\n")
        result["outputs_sha256"] = file_sha256(outputs)
    elif change == "partition":
        prepared = json.loads(local.read_text())
        prepared["partitions"]["humanevalplus"]["development"] = []
        atomic_json(local, prepared)
    else:
        result["status"] = "pass"
    atomic_json(pending / "result.json", result)
    with pytest.raises(ValueError):
        validate_pending(protocol, local, source, pending)
