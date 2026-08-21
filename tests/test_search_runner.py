import json
from types import SimpleNamespace

import pytest

from scripts.architecture_search import query_command
from speck.model import Config, LayerConfig
from speck.search.runner import (
    _artifact_paths,
    _ingest_output,
    _payload,
    _validate_payload,
    evaluate_trial_process,
    recover_results,
    run_search,
    study_lock,
)
from speck.search.scheduler import advance
from speck.search.spec import SearchSettings, objective_names
from speck.search.study import SearchStudy


def settings():
    return SearchSettings.from_dict({
        "format_version": 2,
        "seed": 7,
        "max_architectures": 4,
        "initial_population": 2,
        "population_size": 2,
        "cohort_size": 2,
        "confidence_z": 1.645,
        "space": {
            "min_layers": 1,
            "max_layers": 2,
            "hidden_size_min": 8,
            "hidden_size_max": 12,
            "hidden_size_step": 4,
            "intermediate_size_min": 16,
            "intermediate_size_max": 24,
            "intermediate_size_step": 8,
            "kv_heads": [1, 2],
        },
        "quality": {
            "data_dir": "~/data",
            "batch_tokens": 8,
            "device_batch_size": 1,
            "eval_batch_size": 1,
            "lr": 0.001,
            "min_lr": 0.1,
            "warmup_steps": 1,
            "weight_decay": 0.1,
            "grad_clip": 1.0,
            "optimizer": "adamw",
            "compile": False,
        },
        "validation_slices": [
            {"name": "main", "offset_tokens": 0, "objective": True},
            {"name": "audit", "offset_tokens": 32, "objective": False},
        ],
        "inference": {
            "contexts": [4, 8],
            "warmup_samples": 0,
            "samples": 9,
        },
        "quantization": {"bits": 4, "group_size": 4},
        "rungs": [
            {
                "name": "screen",
                "architecture_limit": 4,
                "seed_count": 1,
                "train_tokens": 16,
                "sequence_length": 4,
                "eval_every_tokens": 8,
                "eval_tokens": 8,
                "inference_samples": 1,
            },
            {
                "name": "verify",
                "architecture_limit": 2,
                "seed_count": 2,
                "train_tokens": 32,
                "sequence_length": 8,
                "eval_every_tokens": 16,
                "eval_tokens": 16,
                "inference_samples": 3,
            },
        ],
    })


def baseline():
    return Config(
        vocab_size=16,
        layers=(LayerConfig(8, 16, 1),),
        head_dim=4,
        max_position_embeddings=10,
    )


def result(value):
    return {
        "objectives": {
            "quality.validation_nll.main": value,
            "memory.kv_cache_bytes_per_token": value,
            "memory.quantized_weight_bytes": value,
            "prefill.ms.context_4": value,
            "decode.ms_per_token.context_4": value,
            "memory.inference_peak_bytes.context_4": value,
            "prefill.ms.context_8": value,
            "decode.ms_per_token.context_8": value,
            "memory.inference_peak_bytes.context_8": value,
        }
    }


def output(trial, attempt_id, status="failed"):
    return {
        "status": status,
        "architecture_id": trial["architecture_id"],
        "trial_id": trial["id"],
        "rung": trial["rung"],
        "seed_index": trial["seed_index"],
        "attempt_id": attempt_id,
        "payload_digest": "test-payload",
        "error_type": "RuntimeError",
        "error": "transient",
    }


def test_trial_payload_resolves_rung_fidelity_and_identity(tmp_path):
    study = SearchStudy(tmp_path / "study.sqlite3")
    study.initialize(
        settings().export(), {"git": {"revision": "test"}}
    )
    advance(study, baseline(), settings())
    trial = study.trials(status="pending")[0]
    attempt_id = study.start_attempt(trial["id"])
    payload = _payload(study, trial, attempt_id, {"kind": "test"}, settings())
    assert payload["trial_id"] == trial["id"]
    assert payload["evaluation_seed"] == trial["seed"]
    assert payload["quality"]["train_tokens"] == 16
    assert payload["quality"]["schedule_steps"] == 4
    assert payload["inference"]["samples"] == 1
    assert payload["validation_slices"][1]["name"] == "audit"
    assert payload["expected_objectives"] == list(objective_names(settings()))
    study.close()


def test_search_lifecycle_and_resume(tmp_path, monkeypatch):
    study = SearchStudy(tmp_path / "study.sqlite3")
    study.initialize(settings().export(), {})

    def evaluate(study, study_dir, trial, tokenizer, search, device):
        attempt_id = study.start_attempt(trial["id"])
        study.complete_attempt(
            trial["id"], attempt_id, result(float(trial["architecture_id"]))
        )
        return True

    monkeypatch.setattr("speck.search.runner.evaluate_trial_process", evaluate)
    run_search(study, tmp_path, baseline(), {}, settings(), "cpu")
    assert study.summary()["trials"] == {"completed": 8}
    assert study.study()["status"] == "completed"
    assert study.study()["recommendations"]["balanced"]["id"]
    assert len(study.rungs(rung=1)) == 2

    run_search(study, tmp_path, baseline(), {}, settings(), "cpu")
    assert len(study.architectures()) == 4
    study.close()


def test_structured_worker_failure_retries_once(tmp_path):
    study = SearchStudy(tmp_path / "study.sqlite3")
    study.initialize(settings().export(), {})
    advance(study, baseline(), settings())
    trial = study.trials(status="pending")[0]

    first_attempt = study.start_attempt(trial["id"])
    assert not _ingest_output(
        study,
        trial,
        first_attempt,
        output(trial, first_attempt),
        settings(),
        "test-payload",
    )
    assert study.trial(trial["id"])["status"] == "pending"

    second_attempt = study.start_attempt(trial["id"])
    assert not _ingest_output(
        study,
        trial,
        second_attempt,
        output(trial, second_attempt),
        settings(),
        "test-payload",
    )
    assert study.trial(trial["id"])["status"] == "failed"
    study.close()


def test_worker_preflight_rejection_does_not_consume_retry(tmp_path):
    study = SearchStudy(tmp_path / "study.sqlite3")
    study.initialize(settings().export(), {})
    advance(study, baseline(), settings())
    trial = study.trials(status="pending")[0]
    attempt_id = study.start_attempt(trial["id"])
    worker_output = output(trial, attempt_id, "interrupted")
    assert not _ingest_output(
        study,
        trial,
        attempt_id,
        worker_output,
        settings(),
        "test-payload",
    )
    assert study.trial(trial["id"])["status"] == "pending"
    assert study.failed_attempt_count(trial["id"]) == 0
    study.close()


def test_worker_output_rejects_mismatched_trial_identity(tmp_path):
    study = SearchStudy(tmp_path / "study.sqlite3")
    study.initialize(settings().export(), {})
    advance(study, baseline(), settings())
    trial = study.trials(status="pending")[0]
    attempt_id = study.start_attempt(trial["id"])
    worker_output = output(trial, attempt_id)
    worker_output["architecture_id"] += 1
    with pytest.raises(ValueError, match="does not match"):
        _ingest_output(
            study,
            trial,
            attempt_id,
            worker_output,
            settings(),
            "test-payload",
        )
    assert study.trial(trial["id"])["status"] == "running"
    study.close()


def test_trial_evaluation_rejects_code_changes(tmp_path, monkeypatch):
    study = SearchStudy(tmp_path / "study.sqlite3")
    study.initialize(settings().export(), {"git": {"revision": "one"}})
    advance(study, baseline(), settings())
    trial = study.trials(status="pending")[0]
    monkeypatch.setattr(
        "speck.search.runner._git_state", lambda: {"revision": "two"}
    )
    with pytest.raises(RuntimeError, match="code changed"):
        evaluate_trial_process(
            study,
            tmp_path,
            trial,
            {},
            settings(),
            "cpu",
        )
    assert study.trial(trial["id"])["status"] == "pending"
    study.close()


def test_trial_launch_failure_returns_attempt_to_pending(tmp_path, monkeypatch):
    git = {"revision": "same"}
    study = SearchStudy(tmp_path / "study.sqlite3")
    study.initialize(settings().export(), {"git": git})
    advance(study, baseline(), settings())
    trial = study.trials(status="pending")[0]
    monkeypatch.setattr("speck.search.runner._git_state", lambda: git)

    def fail_launch(*args, **kwargs):
        raise OSError("cannot launch")

    monkeypatch.setattr("speck.search.runner.subprocess.Popen", fail_launch)
    with pytest.raises(OSError, match="cannot launch"):
        evaluate_trial_process(
            study,
            tmp_path,
            trial,
            {},
            settings(),
            "cpu",
        )
    assert study.trial(trial["id"])["status"] == "pending"
    assert study.failed_attempt_count(trial["id"]) == 0
    assert not study.running_attempts()
    study.close()


def test_payload_digest_detects_fidelity_changes(tmp_path, monkeypatch):
    git = {"revision": "same"}
    study = SearchStudy(tmp_path / "study.sqlite3")
    study.initialize(settings().export(), {"git": git})
    advance(study, baseline(), settings())
    trial = study.trials(status="pending")[0]
    attempt_id = study.start_attempt(trial["id"])
    payload = _payload(study, trial, attempt_id, {}, settings())
    monkeypatch.setattr("speck.search.runner._git_state", lambda: git)
    _validate_payload(payload)
    payload["quality"]["train_tokens"] += 8
    with pytest.raises(ValueError, match="digest"):
        _validate_payload(payload)
    study.close()


def test_completed_output_rejects_boolean_objective(tmp_path):
    study = SearchStudy(tmp_path / "study.sqlite3")
    study.initialize(settings().export(), {})
    advance(study, baseline(), settings())
    trial = study.trials(status="pending")[0]
    attempt_id = study.start_attempt(trial["id"])
    worker_output = output(trial, attempt_id, "completed")
    worker_output["result"] = result(1.0)
    worker_output["result"]["objectives"]["quality.validation_nll.main"] = True
    with pytest.raises(ValueError, match="non-finite"):
        _ingest_output(
            study,
            trial,
            attempt_id,
            worker_output,
            settings(),
            "test-payload",
        )
    study.close()


def test_recovery_ingests_a_completed_trial_artifact(tmp_path):
    git = {"revision": "same"}
    study = SearchStudy(tmp_path / "study.sqlite3")
    study.initialize(settings().export(), {"git": git})
    advance(study, baseline(), settings())
    trial = study.trials(status="pending")[0]
    attempt_id = study.start_attempt(trial["id"])
    payload = _payload(study, trial, attempt_id, {}, settings())
    study.set_attempt_payload(attempt_id, payload["payload_digest"])
    paths = _artifact_paths(tmp_path, trial, attempt_id)
    paths["directory"].mkdir(parents=True)
    paths["input"].write_text(json.dumps(payload), encoding="utf-8")
    worker_output = output(trial, attempt_id, "completed")
    worker_output["payload_digest"] = payload["payload_digest"]
    worker_output["result"] = result(1.0)
    paths["result"].write_text(json.dumps(worker_output), encoding="utf-8")
    assert recover_results(study, tmp_path, settings()) == 0
    assert study.trial(trial["id"])["status"] == "completed"
    study.close()


def test_recovery_handles_a_missing_input_artifact(tmp_path):
    study = SearchStudy(tmp_path / "study.sqlite3")
    study.initialize(settings().export(), {})
    advance(study, baseline(), settings())
    trial = study.trials(status="pending")[0]
    attempt_id = study.start_attempt(trial["id"])
    study.set_attempt_payload(attempt_id, "missing")
    paths = _artifact_paths(tmp_path, trial, attempt_id)
    paths["directory"].mkdir(parents=True)
    paths["result"].write_text("{}", encoding="utf-8")
    assert recover_results(study, tmp_path, settings()) == 0
    assert study.trial(trial["id"])["status"] == "pending"
    assert not study.running_attempts()
    study.close()


def test_cli_status_reads_v2_study(tmp_path, monkeypatch, capsys):
    database = tmp_path / "study.sqlite3"
    study = SearchStudy(database)
    study.initialize(settings().export(), {"device": "cpu"})
    study.close()
    monkeypatch.setattr(
        "scripts.architecture_search.study_dir", lambda name: tmp_path
    )
    query_command(
        SimpleNamespace(
            command="status", study="test", output=None, rung=None
        )
    )
    value = json.loads(capsys.readouterr().out)
    assert value["study"]["status"] == "running"


def test_study_lock_rejects_a_second_coordinator(tmp_path):
    with study_lock(tmp_path):
        with pytest.raises(RuntimeError, match="running coordinator"):
            with study_lock(tmp_path):
                pass
    with study_lock(tmp_path):
        pass
