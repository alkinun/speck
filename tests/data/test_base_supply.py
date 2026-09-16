import json
from pathlib import Path

import pytest

from speck.data.base_supply import analyze, render_markdown
from speck.provenance.io import durable_json, file_sha256


@pytest.fixture
def inputs(tmp_path):
    def save(name, value):
        path = tmp_path / name
        durable_json(path, value)
        return {"path": str(path), "sha256": file_sha256(path)}

    tokenizer = save(
        "tokenizer.json", {"status": "tokenizer_selected_and_frozen", "tokenizer_fingerprint": "t"}
    )
    data = save(
        "data.json",
        {
            "base": {
                "category_percent": dict(
                    web=55, code=15, math=10, synthetic=10, science=5, reference=5
                ),
                "initial_sources": dict(
                    web="FineWeb-Edu",
                    code="restricted Stack v3 under admitted source/language policy",
                    math="FineMath-4+",
                    synthetic="Cosmopedia v2",
                    science="peS2o v3",
                    reference="FineWiki",
                ),
            }
        },
    )
    model = save("model.json", {"flagship": {"base_tokens": 3200, "stretch_tokens": 4000}})
    parent = {"path": "fixture-parent", "sha256": "p"}
    text = save(
        "text.json",
        {
            "training_authority": False,
            "reference_capacity": {"tokens": 100, "documents": 5, "tokenizer": {"sha256": "t"}},
            "analysis": {
                "parent_manifest": parent,
                "exact_reference_overlap": 0,
                "reference_outputs_preserved": True,
            },
        },
    )
    manifest = save(
        "cache-manifest.json",
        {
            "status": "complete_document_token_cache_not_training_view",
            "training_authority": False,
            "token_count": 100,
            "document_count": 5,
            "plan": {
                "source_id": "finemath_4plus",
                "category": "math",
                "stock_result": text,
                "tokenizer_decision": tokenizer,
                "tokenizer": {"sha256": "t"},
                "parent_manifest": parent,
                "expected_tokens": 100,
                "expected_documents": 5,
            },
        },
    )
    cache = save(
        "cache.json",
        {"tokens": 100, "documents": 5, "completed_reopen_pass": True, "manifest": manifest},
    )
    spec = {
        "format": "speck_base_supply_inputs",
        "format_version": 1,
        "data_plan": data,
        "model_plan": model,
        "tokenizer": tokenizer,
        "stocks": [{"category": "math", "text_result": text, "token_result": cache}],
    }
    return Path(save("inputs.json", spec)["path"]), save, spec


def test_demand_exposure_missing_supply_and_no_joint_union_claim(inputs):
    path, _, _ = inputs
    result = analyze(path)
    default, stretch = result["scenarios"]
    assert sum(row["base_processed_token_demand"] for row in default["sources"]) == 3200
    math = next(row for row in default["sources"] if row["category"] == "math")
    assert math["base_processed_token_demand"] == 320
    assert math["optimistic_average_exposure_if_only_this_stock"] == 3.2
    assert stretch["sources"][2]["optimistic_average_exposure_if_only_this_stock"] == 4
    assert default["sources"][0]["source_specific_stock_tokens"] is None
    assert result["globally_unique_tokens"] is None and result["production_ready"] is False
    assert all(row["joint_training_eligible_tokens"] is None for row in default["sources"])
    assert "3.20×" in render_markdown(result)


def test_overlapping_stock_entries_are_rejected(inputs):
    path, _, spec = inputs
    spec["stocks"].append(spec["stocks"][0])
    durable_json(path, spec)
    with pytest.raises(ValueError, match="overlapping banks"):
        analyze(path)


def test_source_substitution_is_not_inferred_from_category(inputs):
    path, save, spec = inputs
    cache = json.loads(Path(spec["stocks"][0]["token_result"]["path"]).read_text())
    manifest = json.loads(Path(cache["manifest"]["path"]).read_text())
    manifest["plan"]["source_id"] = "ultradata_math_l2_preview"
    cache["manifest"] = save("wrong-source.json", manifest)
    spec["stocks"][0]["token_result"] = save("wrong-cache.json", cache)
    durable_json(path, spec)
    with pytest.raises(ValueError, match="lineage differs"):
        analyze(path)


@pytest.mark.parametrize("change", ["counts", "reopen", "tokenizer", "parent"])
def test_inconsistent_receipts_cannot_become_measured_supply(inputs, change):
    path, save, spec = inputs
    cache = json.loads(Path(spec["stocks"][0]["token_result"]["path"]).read_text())
    if change == "counts":
        cache["tokens"] += 1
    elif change == "reopen":
        cache["completed_reopen_pass"] = False
    else:
        manifest = json.loads(Path(cache["manifest"]["path"]).read_text())
        key = "tokenizer" if change == "tokenizer" else "parent_manifest"
        manifest["plan"][key]["sha256"] = "changed"
        cache["manifest"] = save("changed-manifest.json", manifest)
    spec["stocks"][0]["token_result"] = save("changed-cache.json", cache)
    durable_json(path, spec)
    with pytest.raises(ValueError):
        analyze(path)


def test_modified_bound_input_requires_a_successor(inputs):
    path, _, spec = inputs
    Path(spec["model_plan"]["path"]).write_text("{}")
    with pytest.raises(ValueError, match="identity changed"):
        analyze(path)


@pytest.mark.parametrize("changed", [False, True])
def test_legacy_cache_needs_exact_separate_reopen_evidence(inputs, changed):
    path, save, spec = inputs
    cache = json.loads(Path(spec["stocks"][0]["token_result"]["path"]).read_text())
    del cache["completed_reopen_pass"]
    spec["stocks"][0]["token_result"] = save("legacy-cache.json", cache)
    durable_json(path, spec)
    with pytest.raises(ValueError, match="qualification is incomplete"):
        analyze(path)
    spec["stocks"][0]["reopen_receipt"] = save(
        "reopen.json",
        {
            "format": "speck_preparation_recovery_launch",
            "completed_token_stock_post_outage_verification": [
                {
                    "source_id": "finemath_4plus",
                    "documents": 5,
                    "tokens": 100,
                    "manifest_sha256": "other" if changed else cache["manifest"]["sha256"],
                    "payload_hashes_and_index_coverage": "pass",
                }
            ],
        },
    )
    durable_json(path, spec)
    if changed:
        with pytest.raises(ValueError, match="qualification is incomplete"):
            analyze(path)
    else:
        assert analyze(path)["stock_receipts"]["math"]["source_specific_stock_tokens"] == 100
