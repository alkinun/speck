import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.paper_finalist_source_coverage_validate import EXPECTED_SOURCES, validate_result
from speck.paper_finalist_analysis import analyze_finalist, atomic_json, lock_time_to_quality_target
from tests.test_paper_finalist_analysis import (
    contract_path,
    plan_path,
    result,
    write_results,
)


def complete_report():
    pair = {"pair": 0, "seed": 42, "data_token_offset": 0}
    report = result(pair, "dense_global_param_match", 3.0, "2026-09-06T00:00:00+00:00")
    for entry in report["validation_history"]:
        entry["validation_source_losses"] = {
            source: entry["validation_loss"] + index * 0.001
            for index, source in enumerate(EXPECTED_SOURCES)
        }
    report["final_validation"] = report["validation_history"][-1]
    return report


def test_complete_finite_finalist_source_coverage_passes():
    assert validate_result(complete_report()) == {
        "status": "complete_finite_source_coverage",
        "run": None,
        "sources": 11,
        "validation_points": 5,
        "intermediate_minimum_batches_per_source": 27,
        "final_minimum_batches_per_source": 110,
    }


def test_finalist_source_coverage_rejects_one_missing_source():
    report = deepcopy(complete_report())
    report["validation_history"][2]["validation_source_losses"].pop("dclm")
    with pytest.raises(ValueError, match="every expected source"):
        validate_result(report)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_finalist_source_coverage_rejects_non_finite_source(value):
    report = deepcopy(complete_report())
    report["validation_history"][1]["validation_source_losses"]["dclm"] = value
    with pytest.raises(ValueError, match="non-finite source loss"):
        validate_result(report)


def test_frozen_analyzer_alone_does_not_require_expected_source_set(tmp_path):
    controls, candidates = write_results(tmp_path)
    for path in controls + candidates:
        value = json.loads(path.read_text(encoding="utf-8"))
        for entry in value["validation_history"]:
            entry["validation_source_losses"].pop("source-b")
        value["final_validation"] = value["validation_history"][-1]
        atomic_json(path, value)
    target_path = tmp_path / "target.json"
    atomic_json(target_path, lock_time_to_quality_target(plan_path, contract_path, controls))

    report = analyze_finalist(plan_path, contract_path, target_path, controls + candidates)

    assert set(report["source_guardrails"]) == {"source-a"}
    assert report["finalist_language_screen_pass"]


def test_finalist_source_stability_audit_requires_sidecar():
    root = Path(__file__).parents[1]
    audit = json.loads(
        (root / "results" / "Speck-Paper1" / "finalist-source-stability-audit-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["status"] == (
        "training_finiteness_enforced_validation_schedule_complete_source_set_sidecar_required"
    )
    assert audit["validation_schedule"]["sources"] == 11
    assert audit["validation_schedule"]["every_expected_source_has_positive_denominator"]
    assert audit["collector_and_analyzer"]["expected_source_id_set_checked"] is False
    assert audit["collector_and_analyzer"]["synthetic_omission_counterexample_reproduced"]
    assert audit["sidecar"]["required_before_result_interpretation"] is True
    assert audit["decision"]["current_language_sequence_remains_authorized"] is True
    assert audit["decision"]["source_imputation_or_deletion_authorized"] is False
