import copy

import pytest

from scripts.run_preparation_followups import completed_stock, wait_for_source


def valid():
    step = {
        "plan": {"path": "plan", "sha256": "a" * 64},
        "repository_revision": "b" * 40,
        "result_format": "stock",
        "target_tokens": 100,
    }
    result = {
        "plan": step["plan"],
        "repository_revision": step["repository_revision"],
        "format": "stock",
        "training_authority": False,
        "capacity_target_pass": True,
        "storage_gate_pass": True,
        "reference_capacity": {"tokens": 101},
        "analysis": {
            "exact_reference_overlap": 0,
            "reference_outputs_preserved": True,
            "controls": {
                "firewall_exact_control": {"reason": "exact_duplicate"},
                "firewall_near_control": {"reason": "near_duplicate"},
            },
        },
    }
    return step, result


def test_source_completion_requires_bound_result_and_all_gates():
    step, result = valid()
    completed_stock(step, result)
    for key, value in [
        ("plan", {}),
        ("repository_revision", "other"),
        ("capacity_target_pass", False),
        ("storage_gate_pass", False),
        ("training_authority", True),
        ("reference_capacity", {"tokens": 99}),
        ("analysis", {}),
    ]:
        changed = copy.deepcopy(result)
        changed[key] = value
        with pytest.raises(ValueError):
            completed_stock(step, changed)
    result["analysis"]["controls"].pop("firewall_near_control")
    with pytest.raises(ValueError, match="positive controls"):
        completed_stock(step, result)


def test_wait_does_not_treat_running_zero_exit_status_as_completion(monkeypatch):
    states = iter(
        [
            "LoadState=loaded\nActiveState=active\nResult=success\nExecMainStatus=0",
            "LoadState=loaded\nActiveState=inactive\nResult=success\nExecMainStatus=0",
        ]
    )
    monkeypatch.setattr(
        "scripts.run_preparation_followups.subprocess.check_output", lambda *a, **k: next(states)
    )
    slept = []
    monkeypatch.setattr("scripts.run_preparation_followups.time.sleep", slept.append)
    assert wait_for_source("fixture", deadline=float("inf"))["ActiveState"] == "inactive"
    assert slept == [30]


@pytest.mark.parametrize(
    "state",
    [
        "LoadState=not-found\nActiveState=inactive\nResult=success\nExecMainStatus=0",
        "LoadState=loaded\nActiveState=failed\nResult=exit-code\nExecMainStatus=1",
        "LoadState=loaded\nActiveState=inactive\nResult=signal\nExecMainStatus=0",
    ],
)
def test_failed_or_absent_jobs_never_advance(monkeypatch, state):
    monkeypatch.setattr(
        "scripts.run_preparation_followups.subprocess.check_output", lambda *a, **k: state
    )
    with pytest.raises(ValueError):
        wait_for_source("fixture", deadline=float("inf"))


def test_collected_transient_service_uses_publication_without_invented_exit_status(
    monkeypatch, tmp_path
):
    state = "LoadState=not-found\nActiveState=inactive\nResult=success\nExecMainStatus=0"
    monkeypatch.setattr(
        "scripts.run_preparation_followups.subprocess.check_output", lambda *a, **k: state
    )
    result_path = tmp_path / "result.json"
    with pytest.raises(ValueError):
        wait_for_source("fixture", deadline=float("inf"), result_path=result_path)
    result_path.write_text("{}")
    completion = wait_for_source("fixture", deadline=float("inf"), result_path=result_path)
    assert completion["service_exit_status_available"] is False
    assert "ExecMainStatus" not in completion
    # File existence is only a handoff: the bound-result gate still rejects it.
    step, _ = valid()
    with pytest.raises(ValueError):
        completed_stock(step, {})
