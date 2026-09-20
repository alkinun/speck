"""Deterministic, test-backed code-task feasibility for fixed-policy experiments."""

import hashlib
import json

from speck.evaluation.code_runner import run_python


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def run_code_feasibility(tasks, candidates, *, seconds=15, runner=run_python):
    """Grade candidate code against immutable tests; never expose tests to candidate code.

    ``tasks`` contains ``id``, ``test`` and ``entry_point``. ``candidates`` maps task IDs to
    source code. A fresh sandbox invocation is used for every task, which is the reset boundary.
    """

    tasks = list(tasks)
    task_ids = [task.get("id") for task in tasks]
    if not task_ids or any(not isinstance(task_id, str) or not task_id for task_id in task_ids):
        raise ValueError("code feasibility tasks require non-empty IDs")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("code feasibility task IDs must be unique")
    if set(candidates) != set(task_ids):
        raise ValueError("candidate code coverage must exactly match task IDs")
    executions = []
    for task in tasks:
        result = runner(
            candidates[task["id"]],
            task["test"],
            task["entry_point"],
            seconds=seconds,
        )
        executions.append(
            {
                "id": task["id"],
                "status": result.get("status"),
                "wall_seconds": result.get("wall_seconds"),
                "error": result.get("error"),
            }
        )
    passed = sum(row["status"] == "pass" for row in executions)
    return {
        "format": "speck_code_task_feasibility",
        "format_version": 1,
        "status": "fixed_policy_code_verification_only",
        "task_ids": task_ids,
        "executions": executions,
        "summary": {
            "tasks": len(executions),
            "passed": passed,
            "pass_rate": passed / len(executions),
            "timeouts": sum(row["status"] == "timeout" for row in executions),
            "runner_failures": sum(row["status"] == "runner_failure" for row in executions),
        },
        "task_suite_sha256": _digest(
            [
                {"id": task["id"], "test": task["test"], "entry_point": task["entry_point"]}
                for task in tasks
            ]
        ),
        "candidate_sha256": _digest(candidates),
        "boundary": "Fixed-policy code verification only. Tests are immutable and hidden from candidate code; no policy update is performed.",
    }


def verify_code_feasibility(receipt, tasks):
    """Verify task coverage and immutable-test identity before consuming results."""

    if receipt.get("format") != "speck_code_task_feasibility":
        raise ValueError("unsupported code feasibility receipt")
    if receipt.get("status") != "fixed_policy_code_verification_only":
        raise ValueError("code feasibility cannot claim policy training")
    expected = _digest(
        [
            {"id": task["id"], "test": task["test"], "entry_point": task["entry_point"]}
            for task in tasks
        ]
    )
    if receipt.get("task_suite_sha256") != expected:
        raise ValueError("code task suite identity changed")
    if receipt.get("task_ids") != [task["id"] for task in tasks]:
        raise ValueError("code task coverage changed")
    executions = receipt.get("executions")
    summary = receipt.get("summary")
    if not isinstance(executions, list) or not executions or not isinstance(summary, dict):
        raise ValueError("code feasibility execution summary is missing")
    expected_summary = {
        "tasks": len(executions),
        "passed": sum(execution.get("status") == "pass" for execution in executions),
        "pass_rate": sum(execution.get("status") == "pass" for execution in executions)
        / len(executions),
        "timeouts": sum(execution.get("status") == "timeout" for execution in executions),
        "runner_failures": sum(
            execution.get("status") == "runner_failure" for execution in executions
        ),
    }
    if summary != expected_summary:
        raise ValueError("code feasibility summary does not match executions")
    if not receipt.get("boundary", "").startswith("Fixed-policy code verification only"):
        raise ValueError("code feasibility boundary must prohibit policy updates")
    return receipt


def run_code_replay(task, candidates, *, attempts=1, seconds=15, runner=run_python):
    """Run a bounded sequence of attempts with a fresh reset for every attempt.

    The runner receives the immutable test from ``task`` each time. A later successful attempt
    does not erase earlier failures, making recovery cost and reward-hacking behavior visible.
    """

    if not isinstance(task, dict) or not task.get("id") or not task.get("entry_point"):
        raise ValueError("code replay requires a task id and entry point")
    if type(attempts) is not int or not 0 < attempts <= 8:
        raise ValueError("code replay attempts must be an integer in [1, 8]")
    if not isinstance(candidates, (list, tuple)) or not candidates or len(candidates) > attempts:
        raise ValueError("code replay candidates must be a non-empty bounded sequence")
    executions = []
    for code in candidates:
        if not isinstance(code, str):
            raise ValueError("candidate code must be text")
        executions.append(runner(code, task["test"], task["entry_point"], seconds=seconds))
    passed = [execution.get("status") == "pass" for execution in executions]
    return {
        "format": "speck_code_replay_feasibility",
        "format_version": 1,
        "status": "reset_replay_only_no_policy_updates",
        "task_id": task["id"],
        "attempts": len(executions),
        "executions": executions,
        "first_pass_attempt": next(
            (index + 1 for index, value in enumerate(passed) if value), None
        ),
        "task_suite_sha256": _digest(
            {"id": task["id"], "test": task["test"], "entry_point": task["entry_point"]}
        ),
        "outcome_sha256": _digest([execution.get("status") for execution in executions]),
        "boundary": "Every attempt uses a fresh sandbox reset. Failed attempts remain recorded; no policy update is performed.",
    }


def verify_code_replay(receipt, task):
    """Verify replay identity and ensure a success cannot hide prior failures."""

    if receipt.get("format") != "speck_code_replay_feasibility":
        raise ValueError("unsupported code replay receipt")
    if receipt.get("status") != "reset_replay_only_no_policy_updates":
        raise ValueError("code replay cannot claim policy training")
    expected = _digest({"id": task["id"], "test": task["test"], "entry_point": task["entry_point"]})
    if receipt.get("task_suite_sha256") != expected:
        raise ValueError("replay task suite identity changed")
    executions = receipt.get("executions", [])
    if receipt.get("attempts") != len(executions):
        raise ValueError("replay attempt count mismatch")
    if receipt.get("outcome_sha256") != _digest(
        [execution.get("status") for execution in executions]
    ):
        raise ValueError("replay outcomes were changed")
    first_pass = next(
        (
            index + 1
            for index, execution in enumerate(executions)
            if execution.get("status") == "pass"
        ),
        None,
    )
    if receipt.get("first_pass_attempt") != first_pass:
        raise ValueError("replay first-pass accounting mismatch")
    if not receipt.get("boundary", "").startswith("Every attempt uses a fresh sandbox reset"):
        raise ValueError("replay boundary must require fresh resets")
    return receipt
