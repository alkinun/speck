import json
import math
from pathlib import Path

import pytest

from speck.tokenizer_pilot import analyze_tokenizer_pilot, validate_pilot_plan

ROOT = Path(__file__).parents[1]
PLAN = json.loads((ROOT / "research/flagship/tokenizer_pilot_plan.json").read_text())


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
