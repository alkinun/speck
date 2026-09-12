import json
import math

import pytest

from speck.tokenizer_pilot_orchestration import (
    build_completed_run_record,
    macro_bpb,
    orchestration_boundaries,
)


def documents(bpb):
    return [{"document_id": "document", "utf8_bytes": 100, "nll_nats": bpb * math.log(2) * 100}]


def run(tmp_path):
    correction = tmp_path / "correction.json"
    correction.write_text(
        json.dumps(
            {
                "reference": {
                    "fixed_document_tokens": 80,
                    "analytic_flop_target": 8000,
                }
            }
        )
    )
    return {
        "tokenizer_id": "custom",
        "seed": 42,
        "tokenizer": {"model": {"sha256": "tokenizer"}, "vocab_size": 100},
        "model": {
            "sha256": "model",
            "backbone_sha256": "backbone",
            "parameters": 1000,
            "analytic_training_flops_per_token": 100,
        },
        "training_data": {"document_stream_sha256": "documents"},
        "flop_correction": {"path": str(correction), "sha256": "correction"},
        "settings": {
            "batch_tokens": 8,
            "learning_curve_every_steps": 3,
            "checkpoint_every_steps": 4,
        },
        "stops": {
            "fixed_document_tokens": 80,
            "fixed_document_step": 10,
            "fixed_flop_token_stop": 96,
            "final_step": 12,
        },
    }


def categories(bpb):
    return {
        name: documents(bpb)
        for name in ("web", "code", "math", "synthetic", "science", "reference")
    }


def test_orchestration_boundaries_keep_curve_at_fixed_document_endpoint(tmp_path):
    boundaries = orchestration_boundaries(run(tmp_path))

    assert boundaries["learning_curve_steps"] == (3, 6, 9, 10)
    assert boundaries["evaluation_steps"] == (3, 6, 9, 10, 12)
    assert boundaries["checkpoint_steps"] == (4, 8, 10, 12)


def test_completed_record_matches_frozen_analyzer_schema(tmp_path):
    value = run(tmp_path)
    steps = orchestration_boundaries(value)["evaluation_steps"]
    evaluations = {step: categories(1.0 - step / 100) for step in steps}
    timing = {step: float(step * 2) for step in steps}

    result = build_completed_run_record(value, evaluations, timing, 1024)

    assert result["format"] == "speck_tokenizer_pilot_run"
    assert result["fixed_document"]["tokenizer_tokens"] == 80
    assert result["fixed_flop"]["tokenizer_tokens"] == 96
    assert result["fixed_flop"]["analytic_flops"] == 8000
    assert result["learning_curve"][-1]["analytic_flops"] == 8000
    assert result["learning_curve"][-1]["macro_bpb"] == pytest.approx(0.9)
    assert result["fixed_flop"]["categories"] == evaluations[12]


def test_completed_record_rejects_missing_evaluation_or_category(tmp_path):
    value = run(tmp_path)
    steps = orchestration_boundaries(value)["evaluation_steps"]
    evaluations = {step: categories(1.0) for step in steps[:-1]}
    timing = {step: float(step) for step in steps}
    with pytest.raises(ValueError, match="incomplete evaluation"):
        build_completed_run_record(value, evaluations, timing, 1024)

    broken = categories(1.0)
    broken.pop("science")
    with pytest.raises(ValueError, match="six non-empty"):
        macro_bpb(broken)
