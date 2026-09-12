import hashlib
import json
import math
from pathlib import Path

import pytest

from speck.tokenizer_pilot import analyze_tokenizer_pilot, validate_pilot_plan

ROOT = Path(__file__).parents[1]
PLAN = json.loads((ROOT / "research/flagship/tokenizer_pilot_plan.json").read_text())


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_real_pilot_successor_binds_static_corpus_geometry_and_keeps_audit_closed():
    plan = json.loads((ROOT / "research/flagship/tokenizer_pilot_plan_v10.json").read_text())

    assert plan["status"] == "throughput_preflight_pass_run_materializer_pending_D5_unopened"
    assert _sha256(ROOT / plan["supersedes"]["path"]) == plan["supersedes"]["sha256"]
    assert (
        _sha256(ROOT / plan["static_prerequisite"]["path"]) == plan["static_prerequisite"]["sha256"]
    )
    assert (
        _sha256(ROOT / plan["pilot_corpus"]["config"]["path"])
        == plan["pilot_corpus"]["config"]["sha256"]
    )
    for path, digest in plan["implementation"].values():
        assert _sha256(ROOT / path) == digest
    result = plan["pilot_corpus"]["result"]
    assert _sha256(ROOT / result["path"]) == result["sha256"]
    assert plan["pilot_corpus"]["retained_mistral_reference_tokens"] >= 1_200_000_000
    stream = plan["fixed_stream"]
    assert _sha256(ROOT / stream["plan"]["path"]) == stream["plan"]["sha256"]
    for path, digest in stream["implementation"].values():
        assert _sha256(ROOT / path) == digest
    stream_result = stream["result"]
    assert _sha256(ROOT / stream_result["path"]) == stream_result["sha256"]
    assert plan["stopping"]["fixed_document_mistral_tokens"] == 1_200_007_273
    continuation = plan["fixed_flop_materialization"]["continuation"]
    assert _sha256(ROOT / continuation["plan"]["path"]) == continuation["plan"]["sha256"]
    failure = continuation["failed_predecessor"]
    assert _sha256(ROOT / failure["path"]) == failure["sha256"]
    continuation_result = continuation["result"]
    assert _sha256(ROOT / continuation_result["path"]) == continuation_result["sha256"]
    preflight = plan["execution_preflight"]
    assert _sha256(ROOT / preflight["plan"]["path"]) == preflight["plan"]["sha256"]
    assert (
        _sha256(ROOT / preflight["implementation"]["path"]) == preflight["implementation"]["sha256"]
    )
    failed = preflight["failed_predecessor"]
    assert _sha256(ROOT / failed["path"]) == failed["sha256"]
    result = preflight["result"]
    assert _sha256(ROOT / result["path"]) == result["sha256"]
    assert preflight["four_x_stress_gpu_hours"] < plan["gpu_hour_ceiling"]
    for path, digest in continuation["implementation"].values():
        assert _sha256(ROOT / path) == digest
    assert plan["sealed_audit"]["status"] == "unopened"
    assert plan["final_selection_authority"] is False


def test_cuda_preflight_uses_executable_non_underrunning_optimizer_boundaries():
    plan = json.loads((ROOT / "research/flagship/tokenizer_pilot_preflight_v2.json").read_text())
    batch_tokens = plan["settings"]["batch_tokens"]

    assert plan["quality_run_authority"] is False
    assert plan["final_selection_authority"] is False
    assert batch_tokens == 65_536
    assert plan["settings"]["device_batch_size"] == 4
    assert plan["settings"]["accumulation"] == 4
    for tokenizer in plan["tokenizers"]:
        assert tokenizer["fixed_document_aligned_tokens"] % batch_tokens == 0
        assert tokenizer["fixed_flop_aligned_tokens"] % batch_tokens == 0
        assert tokenizer["fixed_document_aligned_tokens"] >= tokenizer["fixed_document_tokens"]
        assert tokenizer["fixed_flop_aligned_tokens"] >= tokenizer["fixed_flop_token_stop"]
        assert tokenizer["run_stop_aligned_tokens"] == max(
            tokenizer["fixed_document_aligned_tokens"],
            tokenizer["fixed_flop_aligned_tokens"],
        )


def _nominations():
    return {
        "format": "speck_tokenizer_static_nomination",
        "status": "fixture_endpoints_reported_no_advancement_authority",
        "selection_authority": False,
        "nominations": [
            {"role": "compression_endpoint", "id": "custom-large"},
            {"role": "compact_endpoint", "id": "custom-small"},
        ],
    }


def _documents(category, bpb, count=20):
    return [
        {
            "document_id": f"{category}-{index}",
            "utf8_bytes": 100,
            "nll_nats": bpb * math.log(2) * 100,
        }
        for index in range(count)
    ]


def _run(tokenizer_id, seed, bpb, *, flops_to_quality, category_override=None):
    categories = {
        category: _documents(
            category, category_override.get(category, bpb) if category_override else bpb
        )
        for category in PLAN["categories"]
    }
    curve_bpb = (
        sum(category_override.get(category, bpb) for category in PLAN["categories"])
        / len(PLAN["categories"])
        if category_override
        else bpb
    )
    return {
        "format": "speck_tokenizer_pilot_run",
        "format_version": 1,
        "status": "complete",
        "tokenizer_id": tokenizer_id,
        "seed": seed,
        "tokenizer_model_sha256": f"tokenizer-{tokenizer_id}",
        "model_manifest_sha256": "same-model",
        "backbone_manifest_sha256": "same-backbone",
        "document_stream_sha256": "same-documents",
        "vocab_size": {"mistral-32k": 32000, "custom-small": 32768, "custom-large": 49152}[
            tokenizer_id
        ],
        "total_parameters": {
            "mistral-32k": 60_000_000,
            "custom-small": 63_000_000,
            "custom-large": 70_000_000,
        }[tokenizer_id],
        "fixed_document": {
            "mistral_reference_tokens": 1_200_000_000,
            "tokenizer_tokens": (1_200_000_000 if tokenizer_id == "mistral-32k" else 1_100_000_000),
            "analytic_flops": 1000,
            "target_analytic_flops": 1000,
            "active_seconds": 100 if tokenizer_id == "mistral-32k" else 90,
            "peak_memory_bytes": 1000,
            "throughput_tokens_per_second": 100,
            "categories": categories,
        },
        "fixed_flop": {
            "mistral_reference_tokens": 1_200_000_000,
            "tokenizer_tokens": 1_000_000_000,
            "analytic_flops": 1000,
            "target_analytic_flops": 1000,
            "active_seconds": 100,
            "peak_memory_bytes": 1000,
            "throughput_tokens_per_second": 100,
            "categories": categories,
        },
        "learning_curve": [
            {"analytic_flops": flops_to_quality, "active_seconds": 80, "macro_bpb": curve_bpb},
            {"analytic_flops": 1000, "active_seconds": 100, "macro_bpb": curve_bpb},
        ],
    }


def _matrix(candidate_bpb=0.99, override=None):
    runs = [
        _run("mistral-32k", 42, 1.0, flops_to_quality=900),
        _run("custom-large", 42, candidate_bpb, flops_to_quality=700, category_override=override),
        _run("custom-small", 42, 1.01, flops_to_quality=800),
    ]
    for seed in (43, 44):
        runs.append(_run("mistral-32k", seed, 1.0, flops_to_quality=900))
        runs.append(
            _run(
                "custom-large",
                seed,
                candidate_bpb,
                flops_to_quality=700,
                category_override=override,
            )
        )
    return runs


def test_matched_candidate_passes_both_views_and_ranks_before_mistral():
    result = analyze_tokenizer_pilot(PLAN, _nominations(), _matrix(), fixture=True)

    assert result["screen"]["selected_custom"] == "custom-large"
    assert result["confirmation"]["eligible_custom"] is True
    assert result["provisional_ranking"] == ["custom-large", "mistral-32k"]
    assert all(
        result["confirmation"]["views"][view]["aggregate_guardrail_pass"]
        and result["confirmation"]["views"][view]["category_guardrails_pass"]
        for view in ("fixed_document", "fixed_flop")
    )
    assert result["audit_handoff"]["opening_authorized"] is False
    assert result["fixed_wall_clock_secondary"]["selection_stopping_view"] is False
    assert set(result["systems"]) == {"custom-large", "custom-small", "mistral-32k"}
    assert result["selection_authority"] is False


def test_category_regression_falls_back_to_mistral():
    override = {category: 0.99 for category in PLAN["categories"]}
    override["code"] = 1.05
    result = analyze_tokenizer_pilot(
        PLAN, _nominations(), _matrix(candidate_bpb=0.99, override=override), fixture=True
    )

    assert result["confirmation"]["eligible_custom"] is False
    assert result["provisional_ranking"][0] == "mistral-32k"


def test_unpaired_documents_and_incomplete_run_matrix_fail_closed():
    runs = _matrix()
    runs[1]["fixed_flop"]["categories"]["web"][0]["document_id"] = "changed"
    with pytest.raises(ValueError, match="not exactly paired"):
        analyze_tokenizer_pilot(PLAN, _nominations(), runs, fixture=True)

    with pytest.raises(ValueError, match="exactly seven"):
        analyze_tokenizer_pilot(PLAN, _nominations(), _matrix()[:-1], fixture=True)


def test_plan_freezes_both_views_bootstrap_and_one_opening_handoff():
    plan = validate_pilot_plan(PLAN)

    assert plan["eligibility"]["required_views"] == ["fixed_document", "fixed_flop"]
    assert plan["statistics"]["bootstrap_replicates"] == 10_000
    assert plan["sealed_audit"] == {
        "identity": "D5_tokenizer",
        "required_finalists": 2,
        "opening": "after provisional ranking only",
        "failure": "mistral-32k",
        "additional_search_after_failure": False,
    }
    assert plan["real_execution"]["status"] == "blocked"
