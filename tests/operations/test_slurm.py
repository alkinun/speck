import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from speck.operations import trainer as slurm_base_train
from speck.operations.slurm import (
    MANDATORY_GPU_HOURS,
    RESERVE_GPU_HOURS,
    TOTAL_GPU_HOURS,
    _budget_lines,
    classify_state,
    daily_summary,
    ingest_sacct,
    load_wave,
    parse_sacct,
    preflight_wave,
    register_manual_reserve,
    render_wave,
    retry_job,
    submit_wave,
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(cwd, *args):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def wave(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    plan = repository / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "format": "speck_program_plan",
                "compute": {
                    "budget_gpu_hours": {
                        "pretraining_ladder": 100,
                        "parent_stable_run": MANDATORY_GPU_HOURS - 100,
                        "reserve": RESERVE_GPU_HOURS,
                    }
                },
            }
        )
    )
    config = tmp_path / "config.json"
    data = tmp_path / "data.json"
    config.write_text('{"frozen": true}\n')
    data.write_text('{"manifest": "frozen"}\n')
    _git(repository, "init")
    _git(repository, "config", "user.email", "fixture@example.invalid")
    _git(repository, "config", "user.name", "Fixture")
    _git(repository, "add", "plan.json")
    _git(repository, "commit", "-m", "fixture")
    commit = _git(repository, "rev-parse", "HEAD")

    identity = [
        {"role": "config", "path": str(config), "sha256": _sha256(config)},
        {"role": "data", "path": str(data), "sha256": _sha256(data)},
    ]
    resources = {
        "nodes": 1,
        "gpus": 1,
        "cpus_per_task": 8,
        "memory_mb": 32_768,
        "walltime_minutes": 60,
        "signal_seconds": 120,
    }
    value = {
        "format": "speck_slurm_wave",
        "format_version": 1,
        "wave_id": "p1-fixture",
        "created_at_utc": "2026-09-08T00:00:00Z",
        "budget": {
            "total_gpu_hours": TOTAL_GPU_HOURS,
            "mandatory_gpu_hours": MANDATORY_GPU_HOURS,
            "reserve_gpu_hours": RESERVE_GPU_HOURS,
        },
        "plan": {"path": str(plan), "sha256": _sha256(plan)},
        "repository": {"path": str(repository), "commit": commit, "require_clean": True},
        "jobs": [
            {
                "id": "screen",
                "budget_line": "pretraining_ladder",
                "kind": "train",
                "allocation": "mandatory",
                "resources": resources,
                "array": {"indices": [0, 1, 2, 3], "max_parallel": 4},
                "command": [
                    "python",
                    "-m",
                    "scripts.slurm_base_train",
                    "{array_index}",
                    "--slurm-requeue-resume",
                ],
                "working_directory": str(repository),
                "identities": identity,
                "max_retries": 1,
                "depends_on": [],
            },
            {
                "id": "flagship",
                "budget_line": "parent_stable_run",
                "kind": "train",
                "allocation": "mandatory",
                "resources": {**resources, "gpus": 4, "walltime_minutes": 120},
                "array": None,
                "command": [
                    "torchrun",
                    "--nproc-per-node=4",
                    "-m",
                    "scripts.slurm_base_train",
                    "experiment",
                    "--slurm-requeue-resume",
                ],
                "working_directory": str(repository),
                "identities": identity,
                "max_retries": 1,
                "depends_on": [],
            },
            {
                "id": "collect",
                "budget_line": "parent_stable_run",
                "kind": "collect",
                "allocation": "mandatory",
                "resources": resources,
                "array": {"indices": [0], "max_parallel": 1},
                "command": ["python", "collect.py", "{array_index}"],
                "working_directory": str(repository),
                "identities": identity,
                "max_retries": 0,
                "depends_on": ["flagship"],
            },
        ],
    }
    path = tmp_path / "wave.json"
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path, value, config, data


def test_wave_validation_is_strict_and_accounts_maximum_attempts_separately(wave):
    path, value, _, _ = wave
    manifest, digest, source, planned = load_wave(path)

    assert source == path
    assert len(digest) == 64
    assert manifest["repository"]["require_clean"] is True
    assert planned == {"mandatory": 25.0, "reserve": 0.0}

    value["unexpected"] = True
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="must contain exactly"):
        load_wave(path)


def test_dependencies_cannot_automate_training_or_scientific_promotion(wave):
    path, value, _, _ = wave
    value["jobs"][1]["depends_on"] = ["screen"]
    path.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="mechanical collect/eval"):
        load_wave(path)


def test_preflight_binds_clean_git_config_and_data(wave):
    path, value, _, data = wave
    manifest, _, _, _ = load_wave(path)
    result = preflight_wave(manifest)
    assert result["identities_verified"] == 6
    assert len(result["git_tree"]) == 40

    repository = Path(value["repository"]["path"])
    (repository / ".opencode-state").write_text("untracked harness state\n")
    preflight_wave(manifest)

    original_plan = (repository / "plan.json").read_text()
    (repository / "plan.json").write_text("tracked change\n")
    with pytest.raises(ValueError, match="clean tracked Git worktree"):
        preflight_wave(manifest)
    (repository / "plan.json").write_text(original_plan)

    data.write_text("changed\n")
    with pytest.raises(ValueError, match="data identity mismatch"):
        preflight_wave(manifest)


def test_render_has_clean_paths_array_and_four_gpu_contract_without_site_guesses(wave, tmp_path):
    path, _, _, _ = wave
    rendered = render_wave(path, tmp_path / "runtime")
    array_script = Path(rendered["scripts"]["screen"]).read_text()
    flagship_script = Path(rendered["scripts"]["flagship"]).read_text()

    assert "#SBATCH --array=0,1,2,3%4" in array_script
    assert "#SBATCH --gres=gpu:1" in array_script
    assert '"${SLURM_ARRAY_TASK_ID}"' in array_script
    assert "#SBATCH --gres=gpu:4" in flagship_script
    assert "#SBATCH --array" not in flagship_script
    assert "#SBATCH --account" not in flagship_script
    assert "#SBATCH --partition" not in flagship_script
    assert "#SBATCH --signal=B:USR1@120" in flagship_script
    assert "scontrol requeue" in flagship_script
    assert "attempt_file" in flagship_script
    assert "SPECK_REQUEUE_SIGNAL_FILE" in flagship_script
    assert "kill -USR1" not in flagship_script
    assert "SPECK_RUN_ID" in flagship_script
    assert "SPECK_MAX_RETRIES=1" in flagship_script
    assert "SPECK_EXPECTED_LOCAL_WORLD_SIZE=4" in flagship_script
    # max-autotune must compile once per wave, not once per preemption.
    assert "export" in flagship_script and "TORCHINDUCTOR_CACHE_DIR" in flagship_script
    assert str(Path(tmp_path / "runtime" / "inductor").resolve()) in flagship_script
    assert Path(rendered["frozen_manifest"]).stat().st_mode & 0o222 == 0
    assert Path(rendered["scripts"]["screen"]).stat().st_mode & 0o222 == 0
    for script in rendered["scripts"].values():
        subprocess.run(["bash", "-n", script], check=True)


def test_slurm_training_rejects_sft_requeue_entrypoint(wave):
    path, value, _, _ = wave
    value["jobs"][0]["command"][2] = "scripts.sft_train"
    path.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="signal-safe Slurm trainer"):
        load_wave(path)


def test_submit_uses_only_mechanical_dependencies_and_refuses_duplicate_or_reserve(wave, tmp_path):
    path, value, _, _ = wave
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=f"{100 + len(calls)};cluster\n")

    record, _ = submit_wave(path, tmp_path / "runtime", runner=runner)
    assert record["jobs"] == {"screen": "101", "flagship": "102", "collect": "103"}
    assert not any("--dependency" in item for item in calls[0][0])
    assert "--dependency=afterok:102" in calls[2][0]
    assert record["budget_commitment_gpu_hours"] == {"mandatory": 25.0, "reserve": 0.0}
    with pytest.raises(ValueError, match="already been submitted"):
        submit_wave(path, tmp_path / "runtime", runner=runner)

    value["jobs"][0]["allocation"] = "reserve"
    value["jobs"][0]["budget_line"] = "reserve"
    value["jobs"][0]["max_retries"] = 0
    reserve = tmp_path / "reserve.json"
    reserve.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="automatic spend refused"):
        submit_wave(reserve, tmp_path / "other", runner=runner)


def test_sacct_ingestion_classifies_and_counts_array_tasks_once(tmp_path):
    text = "\n".join(
        [
            "101|screen|FAILED|1:0|7200|cpu=8,gres/gpu=1|0|2026-09-08T00:00:00|2026-09-08T00:01:00|2026-09-08T02:01:00|",
            "101_0|screen|COMPLETED|0:0|3600|cpu=8,gres/gpu:gh200=1|0|2026-09-08T00:00:00|2026-09-08T00:01:00|2026-09-08T01:01:00|",
            "101_0.batch|batch|COMPLETED|0:0|3600|gres/gpu=1|0||||",
            "101_1|screen|NODE_FAIL|0:0|1800|cpu=8,gres/gpu=1|1|2026-09-08T00:00:00|2026-09-08T00:01:00|2026-09-08T00:31:00|",
            "202|flagship|OUT_OF_MEMORY|0:125|900|cpu=32,gres/gpu=4|0|2026-09-08T00:00:00|2026-09-08T00:01:00|2026-09-08T00:16:00|",
        ]
    )
    rows = parse_sacct(text, {"screen": "101", "flagship": "202"})
    assert [row["job_id"] for row in rows] == ["101_0", "101_1", "202"]
    assert [row["classification"] for row in rows] == [
        "success",
        "mechanical_retryable",
        "resource_configuration_failure",
    ]
    assert sum(row["gpu_hours"] for row in rows) == 2.5
    assert classify_state("COMPLETED", "1:0") == "application_failure"
    assert classify_state("CANCELLED by 1000", "0:0") == "operator_cancelled"

    submission = tmp_path / "submission.json"
    submission.write_text(
        json.dumps(
            {
                "manifest_sha256": "a" * 64,
                "jobs": {"screen": "101", "flagship": "202"},
                "allocations": {"screen": "mandatory", "flagship": "mandatory"},
            }
        )
    )

    def runner(command, **kwargs):
        assert command[0] == "sacct"
        return SimpleNamespace(stdout=text)

    now = datetime(2026, 9, 8, 3, tzinfo=timezone.utc)
    record, record_path = ingest_sacct(submission, tmp_path / "runtime", runner=runner, now=now)
    assert record["jobs"][0]["allocation"] == "mandatory"
    assert Path(record_path).stat().st_mode & 0o222 == 0


def test_retry_is_same_manifest_mechanical_and_bounded(wave, tmp_path):
    path, _, _, _ = wave
    _, digest, _, _ = load_wave(path)
    runtime = tmp_path / "runtime"
    submission = tmp_path / "submission.json"
    submission.write_text(
        json.dumps(
            {
                "manifest_sha256": digest,
                "jobs": {"flagship": "202"},
                "allocations": {"flagship": "mandatory"},
                "attempt_offsets": {"flagship": 0},
            }
        )
    )
    observation = tmp_path / "sacct.json"
    observation.write_text(
        json.dumps(
            {
                "manifest_sha256": digest,
                "submission": str(submission),
                "jobs": [
                    {
                        "job_id": "202",
                        "logical_job_id": "flagship",
                        "classification": "mechanical_retryable",
                        "restart_count": 0,
                    }
                ],
            }
        )
    )
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="303\n")

    record, _ = retry_job(path, "flagship", submission, observation, runtime, runner=runner)
    assert record["attempt_offsets"] == {"flagship": 1}
    assert "--export=ALL,SPECK_RETRY_OFFSET=1" in calls[0]

    submission.write_text(
        json.dumps({**json.loads(submission.read_text()), "attempt_offsets": {"flagship": 1}})
    )
    with pytest.raises(ValueError, match="bound is exhausted"):
        retry_job(path, "flagship", submission, observation, runtime, runner=runner)


@pytest.mark.parametrize("prior_reserve", [0, RESERVE_GPU_HOURS - 13, RESERVE_GPU_HOURS - 12])
def test_manual_reserve_registration_enforces_protected_pool(wave, tmp_path, prior_reserve):
    _, value, _, _ = wave
    authorization = tmp_path / "reserve-approval.json"
    authorization.write_text('{"approved_by": "human operator"}\n')
    authority = {
        "role": "authority",
        "path": str(authorization),
        "sha256": _sha256(authorization),
    }
    for job in value["jobs"]:
        job["allocation"] = "reserve"
        job["budget_line"] = "reserve"
        job["max_retries"] = 0
        job["identities"].append(authority)
    path = tmp_path / "reserve-wave.json"
    path.write_text(json.dumps(value))
    scheduler_jobs = {"screen": "501", "flagship": "502", "collect": "503"}
    runtime = tmp_path / "runtime"
    commitments = runtime / "commitments"
    commitments.mkdir(parents=True)
    (commitments / "prior.json").write_text(
        json.dumps({"gpu_hours": {"mandatory": 0, "reserve": prior_reserve}})
    )
    if prior_reserve + 13 > RESERVE_GPU_HOURS:
        with pytest.raises(ValueError, match="exceed a protected GPU-hour commitment pool"):
            register_manual_reserve(
                path, scheduler_jobs, authorization, _sha256(authorization), runtime
            )
        assert len(list(commitments.glob("*.json"))) == 1
        assert not list((runtime / "submissions").glob("*.json"))
        return

    record, record_path = register_manual_reserve(
        path,
        scheduler_jobs,
        authorization,
        _sha256(authorization),
        runtime,
    )

    assert record["jobs"] == scheduler_jobs
    assert record["budget_commitment_gpu_hours"] == {"mandatory": 0.0, "reserve": 13.0}
    assert record["reserve_spend_automated"] is False
    assert Path(record_path).is_file()
    assert daily_summary(tmp_path / "runtime")["committed_maximum_gpu_hours"] == {
        "mandatory": 0.0,
        "reserve": prior_reserve + 13.0,
    }


def test_daily_summary_keeps_mandatory_and_reserve_usage_separate(tmp_path):
    record_dir = tmp_path / "sacct" / ("a" * 64)
    record_dir.mkdir(parents=True)
    (record_dir / "record.json").write_text(
        json.dumps(
            {
                "collected_at_utc": "2026-09-08T12:00:00+00:00",
                "manifest_sha256": "a" * 64,
                "jobs": [
                    {
                        "job_id": "1",
                        "classification": "success",
                        "gpu_hours": 10.0,
                        "allocation": "mandatory",
                        "ended_at": "2026-09-08T11:00:00+00:00",
                    },
                    {
                        "job_id": "2",
                        "classification": "active",
                        "gpu_hours": 2.0,
                        "allocation": "reserve",
                        "ended_at": None,
                    },
                ],
            }
        )
    )
    summary = daily_summary(tmp_path, day="2026-09-08")
    assert summary["observed_gpu_hours"] == {"mandatory": 10.0, "reserve": 2.0}
    assert summary["mandatory_remaining_gpu_hours"] == MANDATORY_GPU_HOURS - 10
    assert summary["reserve_remaining_gpu_hours"] == RESERVE_GPU_HOURS - 2
    assert summary["ended_today"] == 1


def test_slurm_trainer_requires_job_and_resolves_latest_only_on_retry(tmp_path, monkeypatch):
    cli = SimpleNamespace(
        slurm_requeue_resume=True,
        resume=None,
        output_dir=tmp_path / "checkpoints",
        experiment="experiment",
    )
    configs = {"train": {"run": "fixture", "output_dir": None}}
    with pytest.raises(ValueError, match="requires a Slurm job"):
        slurm_base_train._configure_resume(configs, cli)

    monkeypatch.setenv("SLURM_JOB_ID", "123")
    with pytest.raises(ValueError, match="retry counters"):
        slurm_base_train._configure_resume(configs, cli)
    monkeypatch.setenv("SPECK_MAX_RETRIES", "1")
    slurm_base_train._configure_resume(configs, cli)
    assert cli.resume is None

    monkeypatch.setenv("SPECK_RETRY_OFFSET", "1")
    monkeypatch.setattr(slurm_base_train, "latest", lambda _path: 17)
    slurm_base_train._configure_resume(configs, cli)
    assert cli.resume == 17

    cli.resume = None
    monkeypatch.setenv("SPECK_RETRY_OFFSET", "2")
    with pytest.raises(ValueError, match="bound is exhausted"):
        slurm_base_train._configure_resume(configs, cli)


def _requeue_trainer(monkeypatch, eval_every=0):
    trainer = object.__new__(slurm_base_train.SlurmBaseTrainer)
    trainer.args = SimpleNamespace(
        log_every=100,
        eval_every=eval_every,
        save_every=0,
        warmup_steps=0,
        min_lr=1.0,
        lr_schedule="constant",
        decay_fraction=None,
        grad_clip=1.0,
        lr=1e-3,
        batch_tokens=8,
    )
    trainer.start_step = 0
    trainer.steps = 2
    trainer.schedule_step_offset = 0
    trainer.schedule_steps = 2
    trainer.train_model = object()
    trainer.parameters = ()
    trainer.optimizer = object()
    trainer.train_data = iter(())
    trainer.inputs = trainer.targets = trainer.data_state = object()
    trainer.accumulation = 1
    trainer.distributed = False
    trainer.master = True
    trainer.device = torch.device("cpu")
    trainer.milestones = {}
    trainer.stop_step = None
    trainer.elapsed_optimizer = 0.0
    trainer.elapsed_training = 0.0
    trainer._signal_requested = False
    trainer.interrupted_for_requeue = False
    monkeypatch.setattr(trainer, "_initial_validation", lambda: (1.0, {}, 0, 8))
    monkeypatch.setattr(trainer, "_log_step", lambda *args: None)
    checkpoints = []
    monkeypatch.setattr(
        trainer, "_checkpoint", lambda *args, **kwargs: checkpoints.append((args, kwargs))
    )

    def optimization(*args, **kwargs):
        trainer._signal_requested = True
        zero = torch.tensor(0.0)
        return zero, zero, (object(), object(), object()), 1

    monkeypatch.setattr(slurm_base_train.base_train, "optimization_step", optimization)
    return trainer, checkpoints


def test_slurm_trainer_checkpoints_usr1_at_optimizer_boundary(monkeypatch, tmp_path):
    ready = tmp_path / "signal.ready"
    monkeypatch.setenv("SPECK_REQUEUE_READY_FILE", str(ready))
    trainer, checkpoints = _requeue_trainer(monkeypatch)
    trainer._run_steps()

    assert trainer.completed_step == 1
    assert trainer.interrupted_for_requeue is True
    assert checkpoints[0][0][0] == 1
    assert checkpoints[0][1] == {"partial": True}
    assert len(checkpoints) == 1
    assert json.loads(ready.read_text()) == {"step": 1}


def test_requeue_at_a_validation_step_validates_before_its_checkpoint(monkeypatch):
    trainer, checkpoints = _requeue_trainer(monkeypatch, eval_every=1)
    validated = []
    monkeypatch.setattr(trainer, "_validate", lambda step: validated.append(step) or (0.5, {}, 8))
    trainer._run_steps()

    assert validated == [1]
    step, loss, _, validation_step = checkpoints[0][0][:4]
    assert (step, loss, validation_step) == (1, 0.5, 1)
    assert checkpoints[0][1] == {"partial": True}


def test_slurm_trainer_finishes_missing_final_validation_after_requeue(monkeypatch):
    trainer = object.__new__(slurm_base_train.SlurmBaseTrainer)
    trainer.args = SimpleNamespace()
    trainer.start_step = trainer.completed_step = trainer.steps = 2
    trainer.device = torch.device("cpu")
    trainer.interrupted_for_requeue = False
    trainer.milestones = {}
    monkeypatch.setattr(trainer, "_initial_validation", lambda: (1.0, {}, 1, 8))
    validated = []
    monkeypatch.setattr(trainer, "_validate", lambda step: validated.append(step) or (0.9, {}, 8))
    checkpoints = []
    monkeypatch.setattr(trainer, "_checkpoint", lambda *args: checkpoints.append(args))

    trainer._run_steps()

    assert validated == [2]
    assert checkpoints[0][0] == checkpoints[0][3] == 2


def test_slurm_main_uses_dedicated_requeue_exit_code(monkeypatch):
    monkeypatch.setattr(
        slurm_base_train, "arguments", lambda: SimpleNamespace(experiment="fixture")
    )
    monkeypatch.setattr(slurm_base_train, "load_experiment", lambda *args: {})
    monkeypatch.setattr(slurm_base_train, "train", lambda configs, cli: True)

    with pytest.raises(SystemExit) as raised:
        slurm_base_train.main()
    assert raised.value.code == 99


def test_the_program_plan_defines_the_budget_lines_waves_charge():
    plan = json.loads((Path(__file__).parents[2] / "experiments/main-data/plan.json").read_text())
    lines = _budget_lines(plan)
    assert lines["reserve"] == RESERVE_GPU_HOURS
    assert sum(lines.values()) == TOTAL_GPU_HOURS


def test_distributed_requeue_polls_only_on_shared_check_steps(monkeypatch):
    trainer = object.__new__(slurm_base_train.SlurmBaseTrainer)
    trainer.distributed = True
    polled = []
    monkeypatch.setattr(trainer, "_requeue_requested", lambda: polled.append(1) or True)
    trainer.completed_step = slurm_base_train.REQUEUE_CHECK_STEPS + 1
    assert trainer._optimizer_boundary_stop_requested() is False
    trainer.completed_step = 2 * slurm_base_train.REQUEUE_CHECK_STEPS
    assert trainer._optimizer_boundary_stop_requested() is True
    assert polled == [1]


@pytest.mark.parametrize(
    ("ranks", "requeued", "status"),
    [
        # torchrun reports ranks that exit 99 as its own status 1.
        (': > "${SPECK_REQUEUE_READY_FILE}"; exit 1', True, 0),
        ("exit 1", False, 1),
        (': > "${SPECK_REQUEUE_READY_FILE}"; exit 99', True, 0),
    ],
)
def test_four_gpu_script_requeues_on_the_ready_marker_not_torchrun_status(
    wave, tmp_path, ranks, requeued, status
):
    path, _, _, _ = wave
    rendered = render_wave(path, tmp_path / "runtime")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "scontrol.calls"
    for name, body in (("torchrun", ranks), ("scontrol", f'echo "$@" >> {calls}')):
        (bin_dir / name).write_text(f"#!/usr/bin/env bash\n{body}\n")
        (bin_dir / name).chmod(0o755)
    script = rendered["scripts"]["flagship"]
    text = Path(script).read_text()
    for variable in ("signal_dir", "attempt_dir"):
        line = next(item for item in text.splitlines() if item.startswith(f"readonly {variable}="))
        Path(line.split("=", 1)[1].strip("'")).mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["bash", script],
        env={"PATH": f"{bin_dir}:/usr/bin:/bin", "SLURM_JOB_ID": "7"},
        capture_output=True,
        text=True,
    )

    assert result.returncode == status, result.stderr
    assert calls.is_file() is requeued
    if requeued:
        assert calls.read_text() == "requeue 7\n"
