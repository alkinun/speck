import pytest

from speck.evaluation.rl_code import (
    run_code_feasibility,
    run_code_replay,
    verify_code_feasibility,
    verify_code_replay,
)


TASKS = [
    {"id": "add", "entry_point": "solve", "test": "def check(solve): assert solve(2, 3) == 5"},
    {"id": "broken", "entry_point": "solve", "test": "def check(solve): assert solve(2, 3) == 5"},
]


def fake_runner(code, test, entry_point, *, seconds):
    namespace = {}
    exec(code, namespace)
    exec(test, namespace)
    try:
        namespace["check"](namespace[entry_point])
    except AssertionError:
        return {"status": "failed", "wall_seconds": 0.0}
    return {"status": "pass", "wall_seconds": 0.0}


def test_code_feasibility_is_test_backed_and_verifiable():
    receipt = run_code_feasibility(
        TASKS,
        {"add": "def solve(a, b): return a + b", "broken": "def solve(a, b): return a - b"},
        runner=fake_runner,
    )
    assert receipt["summary"] == {
        "tasks": 2,
        "passed": 1,
        "pass_rate": 0.5,
        "timeouts": 0,
        "runner_failures": 0,
    }
    verify_code_feasibility(receipt, TASKS)


def test_code_feasibility_rejects_changed_hidden_tests():
    receipt = run_code_feasibility(
        TASKS,
        {"add": "def solve(a, b): return a + b", "broken": "def solve(a, b): return a - b"},
        runner=fake_runner,
    )
    changed = [{**TASKS[0], "test": "def check(solve): assert solve(2, 3) == 6"}, TASKS[1]]
    with pytest.raises(ValueError, match="task suite identity"):
        verify_code_feasibility(receipt, changed)


def test_code_replay_preserves_failed_attempts_and_requires_fresh_reset():
    calls = []

    def runner(code, test, entry_point, *, seconds):
        calls.append((code, test, entry_point, seconds))
        return {"status": "pass" if "return a + b" in code else "failed", "wall_seconds": 0.0}

    task = TASKS[0]
    receipt = run_code_replay(
        task,
        ["def solve(a, b): return a - b", "def solve(a, b): return a + b"],
        attempts=2,
        runner=runner,
    )
    assert receipt["first_pass_attempt"] == 2
    assert [call[1] for call in calls] == [task["test"], task["test"]]
    verify_code_replay(receipt, task)


def test_code_replay_rejects_changed_outcomes():
    task = TASKS[0]
    receipt = run_code_replay(
        task,
        ["def solve(a, b): return a + b"],
        runner=lambda code, test, entry_point, *, seconds: {"status": "pass"},
    )
    receipt["executions"][0]["status"] = "failed"
    with pytest.raises(ValueError, match="outcomes"):
        verify_code_replay(receipt, task)


def test_code_feasibility_rejects_tampered_summary():
    receipt = run_code_feasibility(
        TASKS,
        {"add": "def solve(a, b): return a + b", "broken": "def solve(a, b): return a - b"},
        runner=fake_runner,
    )
    receipt["summary"]["pass_rate"] = 1.0
    with pytest.raises(ValueError, match="summary"):
        verify_code_feasibility(receipt, TASKS)
