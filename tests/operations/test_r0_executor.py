"""Qualify bounded execution and failure accounting without allocating flagship tensors."""

import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from speck.model import build_model
from speck.model.architecture import (
    ArchitectureConfig,
    AttentionSpec,
    BlockConfig,
    BlockGroup,
    KimiDeltaAttentionSpec,
    StageConfig,
    SwiGLUSpec,
)
from speck.operations.r0_executor import (
    fingerprint,
    prepare_request,
    run_attempt,
    summarize_attempt,
    supervise,
    validate_settings,
)
from speck.operations.r0_worker import execute_case, failure_kind, synthetic_batch
from speck.provenance.io import durable_json
from speck.provenance.repository import repository_root

ROOT = repository_root()
PLAN = ROOT / "experiments/qualification/plan.json"


def make_tiny_request():
    settings = json.loads(PLAN.read_text())["settings"]
    settings.update(
        loss_backend="torch",
        warmup_steps=1,
        measured_steps=2,
        accumulation=2,
        minimum_free_disk_bytes=1,
    )
    config = ArchitectureConfig(
        (
            BlockGroup(
                BlockConfig(
                    8,
                    (
                        StageConfig(
                            (AttentionSpec(head_dim=4, num_key_value_heads=1, rope_dim=0),)
                        ),
                        StageConfig((SwiGLUSpec(16),)),
                    ),
                )
            ),
        ),
        8,
        vocab_size=19,
        max_position_embeddings=8,
    ).export()
    model = build_model(config, 19)
    value = {
        "case": {
            "id": "cpu-fixture",
            "model": config,
            "model_vocab_size": 19,
            "synthetic_input_vocab_size": 16,
            "sequence_length": 8,
            "instantiated_parameters": model.parameter_count(),
        },
        "settings": settings,
        "world_size": 1,
        "allocated_gpus": 4,
        "r0_gpu_hour_ceiling": 70,
    }
    value["request_sha256"] = fingerprint(value)
    return value


@pytest.fixture
def tiny_request():
    return make_tiny_request()


def test_first_case_binds_actual_model_and_sources_without_execution():
    value = prepare_request(PLAN, "baseline-4096", 4, 4)
    assert value["case"]["model_vocab_size"] == 32003
    assert value["case"]["synthetic_input_vocab_size"] == 32000
    assert value["case"]["instantiated_parameters"] == value["case"]["model"]["expected_parameters"]
    assert value["restart_protocol"] == "fresh_process_next_step_v1"
    assert any(row["path"] == "speck/training/optimizers.py" for row in value["implementation"])
    digest = value.pop("request_sha256")
    assert fingerprint(value) == digest
    with pytest.raises(ValueError, match="unknown checked"):
        prepare_request(PLAN, "hybrid-131072", 1, 4)
    with pytest.raises(ValueError, match="allocated GPUs"):
        prepare_request(PLAN, "baseline-4096", 4, 1)


def test_changed_model_binds_new_identity_and_rejects_invalid_count(tmp_path):
    plan = json.loads(PLAN.read_text())
    model = json.loads((PLAN.parent / plan["model"]).read_text())
    model["expected_parameters"] += 1
    (tmp_path / "model.json").write_text(json.dumps(model))
    path = tmp_path / "plan.json"
    durable_json(path, plan)
    with pytest.raises(ValueError, match="parameter"):
        prepare_request(path, "baseline-4096", 1, 4)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("measured_steps", 0),
        ("timeout_seconds", 3601),
        ("lr", float("nan")),
        ("compile", 1),
        ("deterministic", 1),
        ("resume_tolerance", {"rtol": 0.5, "atol": 0.5}),
    ],
)
def test_invalid_settings_fail_before_execution(tiny_request, key, value):
    tiny_request["settings"][key] = value
    with pytest.raises(ValueError):
        validate_settings(tiny_request["settings"])


def test_synthetic_rank_cursor_replay_preserves_payload_and_lookahead(tiny_request):
    case, settings = tiny_request["case"], tiny_request["settings"]
    a = synthetic_batch(case, settings, 2, 4, 10, "cpu")
    b = synthetic_batch(case, settings, 2, 4, 10, "cpu")
    other = synthetic_batch(case, settings, 3, 4, 10, "cpu")
    assert all(torch.equal(x, y) for x, y in zip(a[:2], b[:2]))
    assert a[2] == b[2] != other[2]
    assert torch.equal(a[0][:, 1:], a[1][:, :-1])
    assert max(a[0].max(), a[1].max()) < 16


def test_actual_cpu_optimization_checkpoint_and_next_step_replay(tiny_request, tmp_path):
    result = execute_case(tiny_request, tmp_path, "cpu")
    assert result["status"] == "bounded_synthetic_checks_pass", result.get("error")
    assert result["backward_optimizer_pass"] and result["checkpoint_next_step_parity_pass"]
    assert result["gpu_fit_pass"] is None and result["four_gpu_ddp_pass"] is None
    assert result["process_restart_parity_pass"] is None
    assert result["cached_generation_parity_pass"] is None
    assert result["synthetic_processed_tokens"] == 2 * 8 * 2
    assert result["input_batches"][-1]["sha256"] == result["input_batches"][-2]["sha256"]
    before = (tmp_path / "rank-0-result.json").read_bytes()
    with pytest.raises(FileExistsError, match="previous worker"):
        execute_case(tiny_request, tmp_path, "cpu")
    assert (tmp_path / "rank-0-result.json").read_bytes() == before


def test_corrupt_optimizer_replay_is_reported_and_original_checkpoint_survives(
    tiny_request, tmp_path, monkeypatch
):
    from speck.operations import r0_worker

    original_load = r0_worker.checkpoint.load

    def changed(*args, **kwargs):
        model, optimizer, metadata = original_load(*args, **kwargs)
        model["embed_tokens.weight"].add_(0.1)
        return model, optimizer, metadata

    monkeypatch.setattr(r0_worker.checkpoint, "load", changed)
    result = execute_case(tiny_request, tmp_path, "cpu")
    assert result["status"] == "numerical_or_parity_failure"
    assert result["checkpoint_next_step_parity_pass"] is None
    assert (tmp_path / "rank-0-checkpoint/complete_000003").exists()


def test_worker_failures_retain_stage_and_do_not_certify_fit(tiny_request, tmp_path):
    tiny_request["case"]["instantiated_parameters"] += 1
    result = execute_case(tiny_request, tmp_path, "cpu")
    assert result["status"] == "execution_failure"
    assert result["stage"] == "construction"
    assert result["gpu_fit_pass"] is None
    assert (tmp_path / "rank-0-result.json").exists()


def test_failure_categories_are_distinct():
    assert failure_kind(torch.cuda.OutOfMemoryError("allocation failed")) == "oom"
    assert failure_kind(ImportError("fla")) == "unsupported_backend"
    assert failure_kind(FloatingPointError("bad gradient")) == "numerical_or_parity_failure"
    assert failure_kind(OSError("disk full")) == "execution_failure"


def test_supervisor_enforces_deadline_and_preserves_log(tmp_path):
    result = supervise(
        [sys.executable, "-c", "import time; print('started', flush=True); time.sleep(30)"],
        tmp_path,
        0.2,
        0.2,
    )
    assert result["termination"] == "timeout"
    assert result["returncode"] != 0
    assert result["supervised_wall_seconds"] < 5
    assert "started" in (tmp_path / "worker.log").read_text()


def test_sigterm_supervisor_stops_worker_and_publishes_interruption(tmp_path):
    # Exercise actual signal delivery rather than mocking process.wait's exception path.
    source = """
import sys
from pathlib import Path
from speck.operations.r0_executor import supervise
from speck.provenance.io import durable_json
p=Path(sys.argv[1])
r=supervise([sys.executable, '-c', 'import time; print("ready", flush=True); time.sleep(30)'], p, 30, 1)
durable_json(p/'supervisor.json', r)
"""
    process = subprocess.Popen([sys.executable, "-c", source, str(tmp_path)])
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            log = tmp_path / "worker.log"
            if log.exists() and "ready" in log.read_text():
                break
            time.sleep(0.02)
        else:
            pytest.fail("supervisor child did not start")
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=5) == 0
        report = json.loads((tmp_path / "supervisor.json").read_text())
        assert report["termination"] == "interrupted"
        assert report["returncode"] != 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_missing_rank_prevents_pass_and_all_allocated_gpus_are_charged(tiny_request, tmp_path):
    tiny_request["world_size"] = 4
    durable_json(
        tmp_path / "rank-0-result.json",
        {
            "rank": 0,
            "world_size": 4,
            "request_sha256": tiny_request["request_sha256"],
            "status": "bounded_synthetic_checks_pass",
        },
    )
    result = summarize_attempt(
        tiny_request,
        tmp_path,
        {"termination": "exited", "returncode": 0, "supervised_wall_seconds": 900},
    )
    assert result["status"] == "attempt_failed_or_incomplete"
    assert result["observed_allocated_gpu_hours"] == 1
    assert result["scheduler_total_gpu_hours"] is None


def test_budget_failure_reserves_cost_and_never_reuses_attempt_directory(
    tiny_request, tmp_path, monkeypatch
):
    from speck.operations import r0_executor

    monkeypatch.setattr(
        r0_executor,
        "supervise",
        lambda *args: {"termination": "timeout", "returncode": -9, "supervised_wall_seconds": 5},
    )
    a = run_attempt(tiny_request, tmp_path, 0)
    b = run_attempt(tiny_request, tmp_path, 0)
    assert a["attempt_directory"] != b["attempt_directory"]
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert len(ledger["attempts"]) == 2
    assert all(
        r["reserved_gpu_hours"] > r["observed_allocated_gpu_hours"] for r in ledger["attempts"]
    )
    with pytest.raises(ValueError, match="accounting changed"):
        run_attempt(tiny_request, tmp_path, 1)
    with pytest.raises(ValueError, match="exceed remaining"):
        run_attempt(tiny_request, tmp_path / "other-ledger", 69.5)
    assert not list((tmp_path / "other-ledger").glob("*/request.json"))


def test_unresolved_previous_reservation_blocks_new_work(tiny_request, tmp_path):
    durable_json(
        tmp_path / "ledger.json",
        {
            "external_prior_gpu_hours": 0,
            "ceiling_gpu_hours": 70,
            "attempts": [{"state": "reserved_or_running"}],
        },
    )
    with pytest.raises(ValueError, match="unresolved prior"):
        run_attempt(tiny_request, tmp_path, 0)


def test_actual_cpu_kda_optimization_and_replay(tiny_request, tmp_path):
    from dataclasses import asdict

    case = tiny_request["case"]
    case["model"]["blocks"][0]["block"]["stages"][0]["branches"][0] = asdict(
        KimiDeltaAttentionSpec(key_head_dim=4, value_head_dim=4, num_key_heads=1, num_value_heads=2)
    )
    case["instantiated_parameters"] = build_model(
        case["model"], case["model_vocab_size"]
    ).parameter_count()
    result = execute_case(tiny_request, tmp_path, "cpu")
    assert result["status"] == "bounded_synthetic_checks_pass", result.get("error")
    assert result["checkpoint_next_step_parity_pass"]


def _cpu_distributed_worker(rank, world_size, tiny_request, directory, rendezvous):
    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world_size)
    try:
        execute_case(tiny_request, directory, "cpu", rank, world_size)
    finally:
        dist.destroy_process_group()


def test_real_two_rank_cpu_ddp_accumulation_and_checkpoint_parity(tiny_request, tmp_path):
    import torch.multiprocessing as mp

    tiny_request["world_size"] = 2
    mp.spawn(
        _cpu_distributed_worker,
        args=(2, tiny_request, str(tmp_path), f"file://{tmp_path}/rendezvous"),
        nprocs=2,
        join=True,
    )
    reports = [json.loads((tmp_path / f"rank-{rank}-result.json").read_text()) for rank in range(2)]
    assert all(r["status"] == "bounded_synthetic_checks_pass" for r in reports), reports
    assert reports[0]["final_model_sha256"] == reports[1]["final_model_sha256"]
    assert reports[0]["input_batches"][0]["sha256"] != reports[1]["input_batches"][0]["sha256"]
    assert all(r["four_gpu_ddp_pass"] is None and r["ddp_rank_weight_parity_pass"] for r in reports)
    assert reports[0]["synthetic_processed_tokens"] == 64


def test_worker_bootstrap_failure_has_a_receipt(tiny_request, tmp_path, monkeypatch):
    from speck.operations.r0_worker import main

    tiny_request["implementation"] = []
    tiny_request.pop("request_sha256")
    tiny_request["request_sha256"] = fingerprint(tiny_request)
    path = tmp_path / "request.json"
    durable_json(path, tiny_request)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(NotImplementedError, match="requires CUDA"):
        main(path)
    result = json.loads((tmp_path / "rank-0-result.json").read_text())
    assert result["stage"] == "bootstrap" and result["status"] == "unsupported_backend"


def test_supervisor_kills_every_independently_grouped_rank(tmp_path):
    command = [
        sys.executable,
        "-c",
        "import os, signal, time; from pathlib import Path; Path(os.environ['R0_TEST_DIRECTORY'], os.environ['RANK']).write_text(str(os.getpid())); signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)",
    ]
    # Each supervised process receives a separate POSIX session, as GPU ranks do in production.
    import os

    old = os.environ.get("R0_TEST_DIRECTORY")
    os.environ["R0_TEST_DIRECTORY"] = str(tmp_path)
    try:
        result = supervise(command, tmp_path, 0.3, 0.1, worker_count=4)
    finally:
        if old is None:
            os.environ.pop("R0_TEST_DIRECTORY")
        else:
            os.environ["R0_TEST_DIRECTORY"] = old
    assert result["termination"] == "timeout"
    assert len(result["rank_returncodes"]) == 4
    assert all(code == -signal.SIGKILL for code in result["rank_returncodes"])
    for rank in range(4):
        pid = int((tmp_path / str(rank)).read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


@pytest.mark.parametrize("corrupt", ["negative_reservation", "changed_result"])
def test_existing_ledger_cannot_hide_cost_or_changed_evidence(
    tiny_request, tmp_path, monkeypatch, corrupt
):
    from speck.operations import r0_executor

    monkeypatch.setattr(
        r0_executor,
        "supervise",
        lambda *args: {"termination": "timeout", "returncode": -9, "supervised_wall_seconds": 5},
    )
    result = run_attempt(tiny_request, tmp_path, 0)
    if corrupt == "negative_reservation":
        ledger_path = tmp_path / "ledger.json"
        ledger = json.loads(ledger_path.read_text())
        ledger["attempts"][0]["reserved_gpu_hours"] = -1
        durable_json(ledger_path, ledger)
    else:
        (Path(result["attempt_directory"]) / "result.json").write_text("{}")
    with pytest.raises(ValueError, match="accounting|result changed"):
        run_attempt(tiny_request, tmp_path, 0)
