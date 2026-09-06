import json
from pathlib import Path

from speck.paper_finalist_analysis import (
    analyze_finalist,
    atomic_json,
    file_sha256,
    lock_time_to_quality_target,
)
from tests.test_paper_finalist_analysis import (
    contract_path,
    plan_path,
    root,
    write_results,
)


def test_finalist_analysis_suppresses_all_time_bounds_after_one_censor(tmp_path):
    controls, candidates = write_results(tmp_path)
    target_path = tmp_path / "target.json"
    atomic_json(target_path, lock_time_to_quality_target(plan_path, contract_path, controls))
    value = json.loads(candidates[0].read_text(encoding="utf-8"))
    for entry in value["validation_history"]:
        entry["validation_loss"] = 4.1
        entry["validation_source_losses"] = {"source-a": 4.1, "source-b": 4.0}
    value["final_validation"] = value["validation_history"][-1]
    atomic_json(candidates[0], value)

    report = analyze_finalist(plan_path, contract_path, target_path, controls + candidates)

    assert report["time_to_quality"]["right_censored_pairs"] == [0]
    assert report["time_to_quality"]["by_data_order"] is None
    assert report["time_to_quality"]["pooled_descriptive"] is None


def test_finalist_secondary_views_preserve_opposed_data_order_effects(tmp_path):
    controls, candidates = write_results(tmp_path)
    target_path = tmp_path / "target.json"
    atomic_json(target_path, lock_time_to_quality_target(plan_path, contract_path, controls))
    for path in candidates:
        value = json.loads(path.read_text(encoding="utf-8"))
        factor = 0.75 if value["pair"]["data_token_offset"] == 0 else 1.5
        for entry in value["validation_history"]:
            entry["steady_training_seconds"] *= factor
            entry["optimizer_seconds"] = entry["steady_training_seconds"] + 3.0
        value["final_validation"] = value["validation_history"][-1]
        atomic_json(path, value)

    report = analyze_finalist(plan_path, contract_path, target_path, controls + candidates)

    time_to_quality = report["time_to_quality"]
    assert time_to_quality["right_censored_pairs"] == []
    assert time_to_quality["by_data_order"]["order_0"]["n"] == 3
    assert time_to_quality["by_data_order"]["order_1610612736"]["n"] == 3
    assert time_to_quality["by_data_order"]["order_0"]["mean"] > 0
    assert time_to_quality["by_data_order"]["order_1610612736"]["mean"] < 0
    assert time_to_quality["pooled_descriptive"]["n"] == 6
    assert time_to_quality["pooled_descriptive"]["mean"] > 0
    for view in ("fixed_analytic_flops", "fixed_steady_training_time"):
        assert report[view]["by_data_order"]["order_0"]["n"] == 3
        assert report[view]["by_data_order"]["order_1610612736"]["n"] == 3
        assert report[view]["pooled_descriptive"]["n"] == 6


def test_finalist_crossed_views_audit_preserves_v2_authority():
    audit_path = (
        Path(__file__).parents[1]
        / "results"
        / "Speck-Paper1"
        / "finalist-crossed-views-audit-v1.json"
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["status"] == (
        "all_inferential_views_order_stratified_censoring_fail_closed_pooled_labels_externalized"
    )
    assert audit["frozen_inputs"]["analysis_plan"]["sha256"] == file_sha256(plan_path)
    assert audit["frozen_inputs"]["analysis_module"]["sha256"] == file_sha256(
        root / "speck" / "paper_finalist_analysis.py"
    )
    assert audit["view_audit"]["time_to_quality"]["any_censor_suppresses_all_bounds"]
    assert audit["view_audit"]["time_to_quality"]["complete_case_bound_forbidden"] is True
    assert audit["reporting_gap"]["statistical_computation_wrong"] is False
    assert audit["decision"]["primary_crossed_factor_inference_qualified"] is True
    assert audit["decision"]["pooled_inference_authorized"] is False
    assert audit["decision"]["current_analyzer_or_plan_modified"] is False
    assert audit["decision"]["analysis_v3_required_before_current_results"] is False
