import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from speck.paper_finalist_systems_telemetry import atomic_json
from speck.paper_finalist_systems_workload import (
    build_trial_plan,
    guarded_run,
    require_output_outside_protected,
)

root = Path(__file__).parents[1]
protocol_path = root / "research" / "paper-1" / "finalist_systems_v2.json"
qualification_path = root / "results" / "Speck-Paper1" / "finalist-qualification-v1.json"


def test_workload_plan_derives_exact_balanced_12_trials_without_checkpoint_access(tmp_path):
    plan = build_trial_plan(protocol_path, qualification_path, tmp_path / "outputs")
    assert plan["status"] == "planned_execution_blocked"
    assert len(plan["trials"]) == 12
    assert [trial["role"] for trial in plan["trials"]] == [
        "control",
        "candidate",
        "candidate",
        "control",
        "candidate",
        "control",
        "control",
        "candidate",
        "control",
        "candidate",
        "candidate",
        "control",
    ]
    assert all(trial["checkpoint_write_authorized"] is False for trial in plan["trials"])
    assert all(trial["execution_authorized"] is False for trial in plan["trials"])
    assert len({trial["checkpoint_directory"] for trial in plan["trials"]}) == 12
    assert not any(Path(trial["output"]).exists() for trial in plan["trials"])


def test_workload_plan_rejects_qualification_relocation(tmp_path):
    value = json.loads(qualification_path.read_text(encoding="utf-8"))
    copied = tmp_path / "finalist-qualification-v1.json"
    atomic_json(copied, value)
    with pytest.raises(ValueError, match="frozen repository"):
        build_trial_plan(protocol_path, copied, tmp_path / "outputs")


def test_output_must_not_contain_or_be_inside_protected_tree(tmp_path):
    protected = tmp_path / "checkpoint"
    protected.mkdir()
    with pytest.raises(ValueError, match="overlaps"):
        require_output_outside_protected(protected / "result.json", [protected])
    with pytest.raises(ValueError, match="overlaps"):
        require_output_outside_protected(tmp_path, [protected])


def test_guarded_run_accepts_unchanged_protected_trees(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    experiment = tmp_path / "experiment"
    checkpoint.mkdir()
    experiment.mkdir()
    (checkpoint / "model.pt").write_bytes(b"model")
    (experiment / "train.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "results" / "trial.json"
    report = guarded_run(
        ["mock-engine"],
        [checkpoint, experiment],
        output,
        runner=lambda _: SimpleNamespace(returncode=0),
    )
    assert report["protected_before"] == report["protected_after"]
    assert report["mutation_detected"] is False
    assert report["kernel_read_only_enforced"] is False


def test_guarded_run_rejects_checkpoint_mutation(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    model = checkpoint / "model.pt"
    model.write_bytes(b"model")

    def mutate(_):
        model.write_bytes(b"changed")
        return SimpleNamespace(returncode=0)

    with pytest.raises(RuntimeError, match="mutated a protected tree"):
        guarded_run(["mock-engine"], [checkpoint], tmp_path / "result.json", runner=mutate)


def test_guarded_run_detects_mutation_even_when_child_raises(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    model = checkpoint / "model.pt"
    model.write_bytes(b"model")

    def mutate_and_raise(_):
        model.write_bytes(b"changed")
        raise ValueError("child failed")

    with pytest.raises(RuntimeError, match="mutated a protected tree") as error:
        guarded_run(
            ["mock-engine"],
            [checkpoint],
            tmp_path / "result.json",
            runner=mutate_and_raise,
        )
    assert isinstance(error.value.__cause__, ValueError)


def test_guarded_run_detects_deleted_protected_tree(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    model = checkpoint / "model.pt"
    model.write_bytes(b"model")

    def delete(_):
        model.unlink()
        checkpoint.rmdir()
        return SimpleNamespace(returncode=0)

    with pytest.raises(RuntimeError, match="mutated a protected tree"):
        guarded_run(["mock-engine"], [checkpoint], tmp_path / "result.json", runner=delete)


def test_guarded_run_propagates_nonzero_without_mutation(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.pt").write_bytes(b"model")
    with pytest.raises(subprocess.CalledProcessError) as error:
        guarded_run(
            ["mock-engine"],
            [checkpoint],
            tmp_path / "result.json",
            runner=lambda _: SimpleNamespace(returncode=7),
        )
    assert getattr(error.value, "returncode", None) == 7
