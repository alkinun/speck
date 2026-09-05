import json
from pathlib import Path

import pytest

from speck.paper_finalist_analysis import (
    analyze_finalist,
    atomic_json,
    collect_run_result,
    file_sha256,
    load_finalist_plan,
    lock_time_to_quality_target,
)

root = Path(__file__).parents[1]
plan_path = root / "research" / "paper-1" / "finalist_analysis_v1.json"
contract_path = root / "research" / "paper-1" / "finalist_materialization_v1.json"
materialization_path = (
    root / "experiments" / "Speck-Paper1-Finalist-131M" / "finalist_materialization.json"
)


def result(pair, arm, final_loss, created_at):
    steps = [0, 5_874, 11_748, 17_622, 23_496]
    seconds = [0.0, 10.0, 20.0, 30.0, 40.0]
    if arm == "five_cache_kda_gqa":
        seconds = [value * 0.8 for value in seconds]
    history = []
    for index, (step, elapsed) in enumerate(zip(steps, seconds)):
        progress = index / (len(steps) - 1)
        loss = 4.0 + progress * (final_loss - 4.0)
        history.append(
            {
                "step": step,
                "global_step": step,
                "training_tokens": step * 65_536,
                "validation_loss": loss,
                "validation_source_losses": {"source-a": loss, "source-b": loss - 0.1},
                "validation_tokens": 19_988_480 if step == 23_496 else 4_997_120,
                "optimizer_seconds": elapsed + 3.0,
                "steady_training_seconds": elapsed,
            }
        )
    geometry = {
        "dense_global_param_match": (153_977_088, 1_301_237_760),
        "five_cache_kda_gqa": (153_958_938, 1_021_601_280),
    }[arm]
    return {
        "format": "speck_paper_finalist_run_result",
        "format_version": 1,
        "status": "complete_qualified",
        "created_at": created_at,
        "paper_id": "speck-paper-1",
        "policy_id": "architecture-promotion-v1",
        "analysis_plan_sha256": file_sha256(plan_path),
        "materialization_contract_sha256": file_sha256(contract_path),
        "materialization_sha256": file_sha256(materialization_path),
        "pair": pair,
        "arm_id": arm,
        "parameters": geometry[0],
        "flops_per_token_at_4096": geometry[1],
        "training_tokens": 1_539_833_856,
        "validation_history": history,
        "final_validation": history[-1],
        "non_finite_steps": 0,
    }


def write_results(tmp_path, candidate_delta=-0.02):
    _, plan, *_ = load_finalist_plan(plan_path, contract_path)
    controls = []
    candidates = []
    for pair in plan["pairs"]:
        control_loss = 3.0 + pair["pair"] * 0.001
        control_path = tmp_path / f"control-{pair['pair']}.json"
        atomic_json(control_path, result(pair, "dense_global_param_match", control_loss, "2026-09-06T00:00:00+00:00"))
        controls.append(control_path)
        candidate_path = tmp_path / f"candidate-{pair['pair']}.json"
        atomic_json(
            candidate_path,
            result(
                pair,
                "five_cache_kda_gqa",
                control_loss + candidate_delta,
                "9999-09-06T00:00:00+00:00",
            ),
        )
        candidates.append(candidate_path)
    return controls, candidates


def test_finalist_analysis_passes_clear_noninferiority(tmp_path):
    controls, candidates = write_results(tmp_path)
    target_path = tmp_path / "target.json"
    atomic_json(target_path, lock_time_to_quality_target(plan_path, contract_path, controls))

    report = analyze_finalist(plan_path, contract_path, target_path, controls + candidates)

    assert report["status"] == "complete_finalist_language_evidence_no_standalone_promotion"
    assert report["fixed_tokens"]["n"] == 6
    assert report["fixed_tokens"]["mean"] == pytest.approx(-0.02)
    assert report["fixed_tokens"]["non_inferiority_pass"]
    assert report["finalist_language_screen_pass"]
    assert not report["time_to_quality"]["right_censored_pairs"]


def test_finalist_analysis_retains_all_pair_censoring(tmp_path):
    controls, candidates = write_results(tmp_path, candidate_delta=0.2)
    target_path = tmp_path / "target.json"
    atomic_json(target_path, lock_time_to_quality_target(plan_path, contract_path, controls))

    report = analyze_finalist(plan_path, contract_path, target_path, controls + candidates)

    assert report["time_to_quality"]["right_censored_pairs"] == list(range(6))
    assert report["time_to_quality"]["paired_relative_improvement"] is None
    assert not report["finalist_language_screen_pass"]


def test_finalist_target_requires_every_control(tmp_path):
    controls, candidates = write_results(tmp_path)
    with pytest.raises(ValueError, match="all six control"):
        lock_time_to_quality_target(plan_path, contract_path, controls[:5] + candidates[:1])


def test_finalist_analysis_requires_every_cell(tmp_path):
    controls, candidates = write_results(tmp_path)
    target_path = tmp_path / "target.json"
    atomic_json(target_path, lock_time_to_quality_target(plan_path, contract_path, controls))
    with pytest.raises(ValueError, match="every frozen cell"):
        analyze_finalist(
            plan_path,
            contract_path,
            target_path,
            controls + candidates[:-1] + candidates[:1],
        )


def test_finalist_plan_matches_materialized_contract():
    _, plan, _, contract, _, manifest = load_finalist_plan(plan_path, contract_path)
    assert plan["pairs"] == [
        {key: pair[key] for key in ("pair", "seed", "data_token_offset")}
        for pair in contract["pairs"]
    ]
    assert manifest["training_authorized"] is False


def test_collect_finalist_result_qualifies_frozen_checkpoint(tmp_path):
    pair = {"pair": 0, "seed": 42, "data_token_offset": 0}
    report = result(pair, "dense_global_param_match", 3.0, "2026-09-06T00:00:00+00:00")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    expected = 23_496
    (checkpoint / f"model_{expected:06d}.pt").write_bytes(b"model")
    (checkpoint / f"optimizer_{expected:06d}.pt").write_bytes(b"optimizer")
    metadata = {
        "step": expected,
        "global_tokens": 1_539_833_856,
        "partial": False,
        "manifest": "b84b09e0b701e35d84487cf6f91e6da9c9fb686b7f6efe67b2e2f5f301fda98e",
        "validation_step": expected,
        "validation_tokens": 19_988_480,
        "validation_history": report["validation_history"],
        "peak_allocated_bytes": 123,
        "resolved": {
            "seed": 42,
            "data_token_offset": 0,
            "train_tokens": 1_539_833_856,
            "batch_tokens": 65_536,
            "sequence_length": 4_096,
            "parameters": 153_977_088,
            "world_size": 1,
        },
    }
    (checkpoint / f"metadata_{expected:06d}.json").write_text(json.dumps(metadata), encoding="utf-8")
    (checkpoint / f"timing_{expected:06d}.json").write_text(
        json.dumps({"optimizer_seconds": 43.0, "steady_training_seconds": 40.0}),
        encoding="utf-8",
    )
    (checkpoint / f"complete_{expected:06d}").write_text("complete\n", encoding="utf-8")
    (checkpoint / "run_summary.json").write_text(
        json.dumps(
            {
                "partial": False,
                "completed_steps": expected,
                "global_tokens": 1_539_833_856,
                "validation_history": report["validation_history"],
            }
        ),
        encoding="utf-8",
    )
    experiment = (
        root
        / "experiments"
        / "Speck-Paper1-Finalist-131M"
        / "runs"
        / "pair-0-seed-42-order-0"
        / "dense_global_param_match"
    )
    collected = collect_run_result(plan_path, contract_path, experiment, checkpoint)
    assert collected["status"] == "complete_qualified"
    assert collected["pair"] == pair
    assert collected["arm_id"] == "dense_global_param_match"
    assert collected["checkpoint"]["model_sha256"] == file_sha256(
        checkpoint / f"model_{expected:06d}.pt"
    )
