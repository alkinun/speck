import hashlib

import numpy as np
import pytest

from speck.model import build_model
from speck.tokenizer_pilot_runs import _fingerprint
from speck.tokenizer_pilot_train import (
    build_pilot_model,
    learning_rate_for_step,
    qualify_checkpoint_resume,
    training_boundaries,
)


def shard(tmp_path, name, values, category):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values, dtype="<u2")
    path.write_bytes(values.tobytes())
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "tokens": len(values),
        "category": category,
    }


def run_manifest(tmp_path):
    model_settings = {
        "blocks": [
            {
                "block": {
                    "hidden_size": 8,
                    "stages": [
                        {
                            "branches": [
                                {
                                    "kind": "attention",
                                    "head_dim": 4,
                                    "num_key_value_heads": 1,
                                    "rope_dim": 0,
                                }
                            ]
                        },
                        {"branches": [{"kind": "swiglu", "intermediate_size": 16}]},
                    ],
                },
                "repeat": 1,
            }
        ],
        "embedding_size": 8,
        "vocab_size": 16,
        "max_position_embeddings": 32,
        "tie_word_embeddings": True,
    }
    model = build_model(model_settings, 16, 1, 2)
    canonical = model.config.settings()
    fixed = shard(tmp_path, "fixed.bin", np.arange(16) % 16, "web")
    continuation = shard(tmp_path, "continuation.bin", np.arange(16, 65) % 16, "web")
    run = {
        "format": "speck_tokenizer_pilot_corrected_screen_run",
        "format_version": 2,
        "status": "corrected_materialized_not_started_execution_blocked",
        "run_id": "fixture-run",
        "tokenizer_id": "fixture-tokenizer",
        "repository_revision": "fixture",
        "model": {
            "settings": canonical,
            "sha256": _fingerprint(canonical),
            "parameters": model.parameter_count(),
            "analytic_training_flops_per_token": model.flops_per_token(2),
        },
        "tokenizer": {
            "model": {"path": "tokenizer.model", "sha256": "tokenizer"},
            "vocab_size": 16,
            "bos_token_id": 1,
            "eos_token_id": 2,
        },
        "training_data": {
            "fixed_stream": {"path": "fixed.json", "sha256": "fixed"},
            "continuation": {"path": "continuation.json", "sha256": "continuation"},
            "fixed_shards": [fixed],
            "continuation_shards": [continuation],
        },
        "evaluation": {"categories": []},
        "flop_correction": {"path": "correction.json", "sha256": "correction"},
        "settings": {
            "sequence_length": 2,
            "device_batch_size": 2,
            "batch_tokens": 8,
            "accumulation": 2,
            "optimizer": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.1,
            "grad_clip": 1.0,
            "lr_schedule": "cosine",
            "warmup_steps": 1,
            "min_lr": 0.1,
            "loss_backend": "torch",
            "checkpoint_every_steps": 2,
            "learning_curve_every_steps": 1,
        },
        "stops": {
            "fixed_document_tokens": 16,
            "fixed_document_step": 2,
            "fixed_flop_token_stop": 32,
            "run_stop_aligned_tokens": 32,
            "final_step": 4,
        },
        "seed": 42,
        "authority": {
            "v2_run_materialization": True,
            "screen_execution": False,
            "confirmation_execution": False,
            "D5_opening": False,
            "final_selection": False,
            "flagship_training": False,
        },
    }
    run["run_fingerprint"] = _fingerprint(run)
    return run


def test_boundaries_include_intervals_fixed_document_and_final(tmp_path):
    run = run_manifest(tmp_path)

    boundaries = training_boundaries(run)

    assert boundaries["learning_curve_steps"] == (1, 2, 3, 4)
    assert boundaries["checkpoint_steps"] == (2, 4)
    assert learning_rate_for_step(run, 0) == run["settings"]["learning_rate"]
    assert learning_rate_for_step(run, 3) == pytest.approx(0.0001)


def test_model_construction_checks_manifest_accounting(tmp_path):
    run = run_manifest(tmp_path)
    model = build_pilot_model(run, "cpu")
    assert model.parameter_count() == run["model"]["parameters"]

    run["model"]["parameters"] += 1
    run["run_fingerprint"] = _fingerprint(
        {key: value for key, value in run.items() if key != "run_fingerprint"}
    )
    with pytest.raises(ValueError, match="constructed"):
        build_pilot_model(run, "cpu")


def test_two_step_checkpoint_resume_is_exact(tmp_path):
    result = qualify_checkpoint_resume(run_manifest(tmp_path), tmp_path / "qualification")

    assert result["status"] == "two_step_checkpoint_resume_exact_no_screen_authority"
    assert result["equivalence"] == {
        "loss": True,
        "model": True,
        "optimizer": True,
        "data_cursor": True,
    }
    assert result["qualification_optimizer_boundaries_executed"] == 3
    assert result["scientific_model_outputs_created"] == 0
    assert result["screen_execution_authority"] is False
