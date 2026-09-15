"""Check persisted state in new worker processes and reject unqualified restart inputs."""

import json
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from speck.operations.r0_executor import prepare_request, run_attempt
from speck.operations.r0_replay import verified_reference
from speck.operations.r0_worker import execute_case
from speck.provenance.io import durable_json
from speck.provenance.repository import repository_root
from tests.operations.test_r0_executor import make_tiny_request

ROOT = repository_root()


@pytest.fixture
def tiny_request():
    return make_tiny_request()


def _phase_worker(rank, world_size, request, base, phase):
    import torch.distributed as dist

    torch.set_num_threads(1)
    output = Path(base) / phase
    if world_size > 1:
        dist.init_process_group(
            "gloo", init_method=(output / "rendezvous").as_uri(), rank=rank, world_size=world_size
        )
    try:
        execute_case(
            request,
            output,
            "cpu",
            rank,
            world_size,
            restart_from=(Path(base) / "initial" / f"rank-{rank}-checkpoint")
            if phase == "restart"
            else None,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def phase(request, base, name, workers):
    (base / name).mkdir()
    mp.spawn(_phase_worker, args=(workers, request, str(base), name), nprocs=workers, join=True)
    return [
        json.loads((base / name / f"rank-{rank}-result.json").read_text())
        for rank in range(workers)
    ]


@pytest.mark.parametrize("workers", [1, 2])
def test_fresh_process_optimizer_cursor_rng_and_distributed_parity(tiny_request, tmp_path, workers):
    tiny_request["restart_protocol"] = "fresh_process_next_step_v1"
    tiny_request["world_size"] = workers
    before = phase(tiny_request, tmp_path, "initial", workers)
    assert all(r["status"] == "restart_reference_ready" for r in before), before
    after = phase(tiny_request, tmp_path, "restart", workers)
    assert all(r["status"] == "bounded_synthetic_checks_pass" for r in after), after
    assert {r["pid"] for r in before}.isdisjoint(r["pid"] for r in after)
    for a, b in zip(before, after, strict=True):
        assert b["process_restart_parity_pass"] and b["rng_continuation_pass"]
        assert b["reference_producer_pid"] == a["pid"]
        assert b["checkpoint_next_step_parity_pass"] is None  # No in-process result substituted.
        assert b["gpu_fit_pass"] is None
        assert a["input_batches"][-1]["sha256"] == b["input_batches"][0]["sha256"]
        assert a["steps"][-1]["loss"] == b["steps"][0]["loss"]
    if workers > 1:
        assert len({r["final_model_sha256"] for r in after}) == 1


def test_restart_rejects_corrupted_rng_before_restoring_model(tiny_request, tmp_path):
    tiny_request["restart_protocol"] = "fresh_process_next_step_v1"
    before = phase(tiny_request, tmp_path, "initial", 1)
    assert before[0]["status"] == "restart_reference_ready"
    (tmp_path / "initial/rank-0-checkpoint/rng.pt").write_bytes(b"corrupted")
    after = phase(tiny_request, tmp_path, "restart", 1)
    assert after[0]["status"] == "execution_failure"
    assert "RNG payload identity" in after[0]["error"]["message"]
    assert after[0]["process_restart_parity_pass"] is None


def test_restart_requires_completed_producer_reference(tiny_request, tmp_path):
    tiny_request["restart_protocol"] = "fresh_process_next_step_v1"
    before = phase(tiny_request, tmp_path, "initial", 1)
    directory = tmp_path / "initial/rank-0-checkpoint"
    reference = json.loads((directory / "restart-reference.json").read_text())
    reference["next_step_loss"] += 1
    durable_json(directory / "restart-reference.json", reference)
    with pytest.raises(ValueError, match="completed producer"):
        verified_reference(directory, tiny_request, 0, 1)
    assert before[0]["status"] == "restart_reference_ready"


def test_two_phase_plan_binding_and_budget_reserves_both_generations(
    tiny_request, tmp_path, monkeypatch
):
    from speck.operations import r0_executor

    request = prepare_request(
        ROOT / "research/flagship/r0_execution_preparation_v2.json", "hybrid-4096", 4, 4
    )
    assert request["restart_protocol"] == "fresh_process_next_step_v1"
    tiny_request["restart_protocol"] = request["restart_protocol"]
    calls = []

    def fake(command, directory, *args):
        calls.append(directory.name)
        status = (
            "restart_reference_ready"
            if directory.name == "initial"
            else "bounded_synthetic_checks_pass"
        )
        durable_json(
            directory / "rank-0-result.json",
            {
                "rank": 0,
                "world_size": 1,
                "request_sha256": tiny_request["request_sha256"],
                "status": status,
            },
        )
        return {"termination": "exited", "returncode": 0, "supervised_wall_seconds": 1}

    monkeypatch.setattr(r0_executor, "supervise", fake)
    result = run_attempt(tiny_request, tmp_path, 0)
    assert calls == ["initial", "restart"]
    assert result["status"] == "bounded_fresh_process_checks_pass"
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert ledger["attempts"][0]["reserved_gpu_hours"] == pytest.approx(2 * 910 * 4 / 3600)


def test_initial_failure_prevents_restart_and_retains_full_reservation(
    tiny_request, tmp_path, monkeypatch
):
    from speck.operations import r0_executor

    tiny_request["restart_protocol"] = "fresh_process_next_step_v1"
    calls = []

    def failed(command, directory, *args):
        calls.append(directory.name)
        return {"termination": "timeout", "returncode": -9, "supervised_wall_seconds": 1}

    monkeypatch.setattr(r0_executor, "supervise", failed)
    result = run_attempt(tiny_request, tmp_path, 0)
    assert calls == ["initial"]
    assert result["status"] == "attempt_failed_or_incomplete"
    assert not (Path(result["attempt_directory"]) / "restart").exists()


def test_fresh_process_kda_model_and_optimizer_parity(tiny_request, tmp_path):
    from dataclasses import asdict

    from speck.model import build_model
    from speck.model.architecture import KimiDeltaAttentionSpec

    case = tiny_request["case"]
    case["model"]["blocks"][0]["block"]["stages"][0]["branches"][0] = asdict(
        KimiDeltaAttentionSpec(key_head_dim=4, value_head_dim=4, num_key_heads=1, num_value_heads=2)
    )
    case["instantiated_parameters"] = build_model(
        case["model"], case["model_vocab_size"]
    ).parameter_count()
    tiny_request["restart_protocol"] = "fresh_process_next_step_v1"
    before = phase(tiny_request, tmp_path, "initial", 1)
    after = phase(tiny_request, tmp_path, "restart", 1)
    assert before[0]["status"] == "restart_reference_ready", before
    assert after[0]["status"] == "bounded_synthetic_checks_pass", after
    assert after[0]["process_restart_parity_pass"] and after[0]["rng_continuation_pass"]
