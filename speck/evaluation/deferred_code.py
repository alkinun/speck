"""Finalize deferred code scores on a host with the required isolated runner."""

import argparse
import copy
import json
from pathlib import Path

from speck.evaluation.capability import selected_rows, verify_scorers
from speck.evaluation.code_runner import check_sandbox, run_python
from speck.provenance.io import atomic_json, file_sha256


def portable_prepared(value):
    value = copy.deepcopy(value)
    for row in value["benchmarks"]:
        row.pop("path")
    return value


def validate_pending(protocol_path, prepared_path, source_prepared_path, pending):
    result = json.loads((pending / "result.json").read_text())
    prepared = json.loads(Path(prepared_path).read_text())
    source_prepared = json.loads(Path(source_prepared_path).read_text())
    if (
        result.get("status") != "code_grading_pending"
        or result.get("defer_code_grading") is not True
    ):
        raise ValueError("a complete generation run with deferred code grading is required")
    if (
        result["protocol_sha256"] != file_sha256(protocol_path)
        or prepared["protocol_sha256"] != result["protocol_sha256"]
    ):
        raise ValueError("protocol identity changed")
    if result["prepared_sha256"] != file_sha256(source_prepared_path) or portable_prepared(
        prepared
    ) != portable_prepared(source_prepared):
        raise ValueError("prepared task identity changed; only paths may be relocated")
    outputs_path = pending / "outputs.jsonl"
    if file_sha256(outputs_path) != result["outputs_sha256"]:
        raise ValueError("generated outputs changed")
    outputs = [json.loads(line) for line in outputs_path.read_text().splitlines()]
    benchmark = next(row for row in prepared["benchmarks"] if row["id"] == "humanevalplus")
    docs = selected_rows(prepared, benchmark, result["partition"], result["limit"])
    code = [row for row in outputs if row["benchmark"] == "humanevalplus"]
    identities = [row["task_id"] for row in code]
    if len(identities) != len(set(identities)) or set(identities) != {
        doc["speck_task_id"] for doc in docs
    }:
        raise ValueError("generated code task coverage changed")
    if any(
        row["execution"] != {"status": "pending_local_grading"} or row["metrics"] for row in code
    ):
        raise ValueError("code records must be ungraded")
    return result, docs, code


def grade(protocol_path, prepared_path, source_prepared_path, pending, output):
    from evalplus.sanitize import sanitize

    check_sandbox()
    pending, output = Path(pending), Path(output)
    result, docs, rows = validate_pending(
        protocol_path, prepared_path, source_prepared_path, pending
    )
    protocol = json.loads(Path(protocol_path).read_text())
    scorers = verify_scorers(protocol)
    if run_python("def f(): return 7", "def check(f): assert f() == 7", "f")["status"] != "pass":
        raise RuntimeError("local code sandbox control failed")
    output.mkdir(parents=True, exist_ok=False)
    tasks = {doc["speck_task_id"]: doc for doc in docs}
    executions = []
    for row in rows:
        doc = tasks[row["task_id"]]
        code = sanitize(
            row["response"] if result["chat"] else doc["prompt"] + row["response"],
            entrypoint=doc["entry_point"],
        )
        execution = run_python(code, doc["test"], doc["entry_point"])
        executions.append({"task_id": row["task_id"], **execution})
        atomic_json(output / "executions.json", executions)
    if not executions:
        raise ValueError("empty code evaluation")
    result["results"]["humanevalplus"]["metrics"] = {
        "compiled_plus_pass@1": sum(row["status"] == "pass" for row in executions) / len(executions)
    }
    result.update(
        status="pass",
        defer_code_grading=False,
        code_grading={
            "source_result_sha256": file_sha256(pending / "result.json"),
            "source_outputs_sha256": file_sha256(pending / "outputs.jsonl"),
            "source_prepared_sha256": file_sha256(source_prepared_path),
            "local_prepared_sha256": file_sha256(prepared_path),
            "scorers": scorers,
            "implementation_sha256": file_sha256(__file__),
            "executions_sha256": file_sha256(output / "executions.json"),
            "boundary": "Only code grading completed here; all generation and other metrics are preserved from the hashed source run.",
        },
    )
    atomic_json(output / "result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "prepared", "source-prepared", "pending", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            grade(args.protocol, args.prepared, args.source_prepared, args.pending, args.output),
            indent=2,
        )
    )
