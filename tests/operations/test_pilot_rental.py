import json
import sys
import time

import pytest

from speck.operations import pilot_rental as rental
from speck.provenance.io import durable_json, file_sha256


def empty_ledger(prior=0):
    return {
        "format": rental.FORMAT,
        "prior_gpu_hours": prior,
        "ceiling_gpu_hours": 50,
        "attempts": [],
    }


@pytest.mark.parametrize("prior", [-1, float("nan"), float("inf"), True, 44.01])
def test_invalid_or_excessive_prior_cost_refuses_launch(tmp_path, prior):
    with pytest.raises(ValueError):
        rental.check_ledger(empty_ledger(prior), prior, tmp_path)


def test_reservations_charge_failed_attempts_and_detect_changed_evidence(tmp_path):
    ledger = empty_ledger(38)
    report = {"status": "failed", "observed_gpu_hours": 0.01}
    path = tmp_path / "attempt/result.json"
    durable_json(path, report)
    ledger["attempts"].append(
        {
            "id": "attempt",
            "state": "failed",
            "reserved_gpu_hours": 6,
            "observed_gpu_hours": 0.01,
            "result_sha256": file_sha256(path),
        }
    )
    assert rental.check_ledger(ledger, 38, tmp_path) == 44
    with pytest.raises(ValueError, match="remaining pilot budget"):
        rental.check_ledger(ledger, 38.01, tmp_path)
    ledger["attempts"][0]["reserved_gpu_hours"] = 0.01
    with pytest.raises(ValueError, match="differs from retained result"):
        rental.check_ledger(ledger, 38, tmp_path)
    ledger["attempts"][0]["reserved_gpu_hours"] = 6
    path.write_text("{}")
    with pytest.raises(ValueError, match="result changed"):
        rental.check_ledger(ledger, 38, tmp_path)


def test_unresolved_attempt_blocks_retry_and_external_cost_cannot_decrease(tmp_path):
    ledger = empty_ledger(2)
    assert rental.check_ledger(ledger, 3, tmp_path) == 3
    with pytest.raises(ValueError, match="accounting changed"):
        rental.check_ledger(ledger, 1, tmp_path)
    ledger["attempts"].append({"state": "reserved_or_running"})
    with pytest.raises(ValueError, match="unresolved attempt"):
        rental.check_ledger(ledger, 3, tmp_path)


def test_packet_verification_detects_payload_tampering_and_checkout_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(rental, "clean_commit", lambda: "source")
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"frozen")
    manifest = {
        "format": rental.FORMAT,
        "commit": "source",
        "files": [{"path": "payload.bin", "bytes": 6, "sha256": file_sha256(payload)}],
    }
    manifest["sha256"] = rental.fingerprint(manifest)
    durable_json(tmp_path / "packet.json", manifest)
    assert rental.verify_packet(tmp_path) == manifest
    payload.write_bytes(b"edited")
    with pytest.raises(ValueError, match="input changed"):
        rental.verify_packet(tmp_path)
    payload.write_bytes(b"frozen")
    monkeypatch.setattr(rental, "clean_commit", lambda: "other")
    with pytest.raises(ValueError, match="checkout differs"):
        rental.verify_packet(tmp_path)


def test_binding_preserves_scientific_settings_and_launches_only_development(tmp_path):
    root, output = tmp_path / "packet", tmp_path / "attempt"
    from speck.config import load_experiment
    from speck.training.base import arguments

    configs = load_experiment("experiments/pilot", "model", "data", "train", "tokenizer")
    durable_json(root / "configs.json", configs)
    durable_json(
        root / "prepared.json",
        {"benchmarks": [{"path": "evaluation/code.jsonl"}], "partitions": {"keep": "unchanged"}},
    )
    rental.bind(root, output)
    bound = load_experiment(output / "experiment", "model", "data", "train", "tokenizer")
    assert bound["model"] == configs["model"]
    assert bound["train"] == {**configs["train"], "wandb_project": None}
    assert bound["data"]["mixture"] == configs["data"]["mixture"]
    assert json.loads((output / "prepared.json").read_text())["partitions"] == {"keep": "unchanged"}
    phases = dict(rental.commands(root, output))
    train = arguments(phases["train"][3:])
    assert train.no_compile and train.device == "cuda" and train.resume is None
    assert train.output_dir == output / "checkpoints"
    assert phases["export"][phases["export"].index("--step") + 1] == "800"
    dev = phases["development"]
    assert dev[dev.index("--partition") + 1] == "development"
    assert dev[dev.index("--limit") + 1] == "0"
    assert "--chat" not in dev


def test_phase_failure_stops_before_next_command_and_preserves_logs(tmp_path):
    marker = tmp_path / "unexpected"
    phases = [
        ("train", [sys.executable, "-c", "raise SystemExit(7)"]),
        ("export", [sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"]),
    ]
    with pytest.raises(RuntimeError, match="train"):
        rental.execute_phases(phases, tmp_path, time.monotonic() + 30)
    assert not marker.exists()
    assert (tmp_path / "logs/train/worker.log").is_file()
    assert json.loads((tmp_path / "phases.json").read_text())["train"]["returncode"] == 7


def test_one_shared_deadline_bounds_hanging_phase(tmp_path, monkeypatch):
    monkeypatch.setattr(rental, "GRACE_SECONDS", 0.05)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="timeout"):
        rental.execute_phases(
            [("train", [sys.executable, "-c", "import time; time.sleep(60)"])],
            tmp_path,
            started + 5.3,
        )
    assert time.monotonic() - started < 5
    assert json.loads((tmp_path / "phases.json").read_text())["train"]["termination"] == "timeout"


def test_phase_log_directory_does_not_precreate_export_destination(tmp_path):
    target = tmp_path / "export"
    rental.execute_phases(
        [("export", [sys.executable, "-c", f"import os; os.mkdir({str(target)!r})"])],
        tmp_path,
        time.monotonic() + 30,
    )
    assert target.is_dir()


def test_failed_run_is_accounted_and_environment_restored(tmp_path, monkeypatch):
    monkeypatch.setattr(
        rental, "verify_packet", lambda root: {"sha256": "packet", "commit": "source"}
    )
    monkeypatch.setattr(rental, "bind", lambda *args: None)
    monkeypatch.setenv("WANDB_MODE", "test-original")

    def fail(*args):
        raise RuntimeError("deliberate failure")

    monkeypatch.setattr(rental, "execute_phases", fail)
    with pytest.raises(RuntimeError, match="deliberate failure"):
        rental.run(tmp_path / "packet", tmp_path / "ledger", 0)
    import os

    assert os.environ["WANDB_MODE"] == "test-original"
    ledger = json.loads((tmp_path / "ledger/ledger.json").read_text())
    assert ledger["attempts"][0]["state"] == "failed"
    assert rental.check_ledger(ledger, 0, tmp_path / "ledger") == 6
    result = json.loads(
        (tmp_path / "ledger" / ledger["attempts"][0]["id"] / "result.json").read_text()
    )
    assert "deliberate failure" in result["error"]


def test_deferred_grading_is_explicit_in_all_relevant_commands(tmp_path):
    phases = dict(
        rental.commands(tmp_path / "packet", tmp_path / "attempt", defer_code_grading=True)
    )
    assert all(
        "--defer-code-grading" in phases[name] for name in ("preflight", "graders", "development")
    )
    assert all("--defer-code-grading" not in phases[name] for name in ("train", "export"))
    assert "--partition" in phases["development"] and "development" in phases["development"]


def test_single_gpu_development_does_not_initialize_accidental_distributed_state(tmp_path):
    command = dict(rental.commands(tmp_path / "packet", tmp_path / "attempt"))["development"]
    prefix = command[: command.index(sys.executable)]
    probe = [
        *prefix,
        sys.executable,
        "-c",
        "import os; assert not ({'RANK', 'LOCAL_RANK', 'WORLD_SIZE'} & os.environ.keys())",
    ]
    result = rental.execute_phases(
        [("development", probe)], tmp_path / "probe", time.monotonic() + 30
    )
    assert result["development"]["returncode"] == 0
