import json
from copy import deepcopy
from pathlib import Path

import pytest

from speck.paper_baseline_analysis import (
    analyze_baselines,
    atomic_json,
    collect_run_result,
    file_sha256,
    load_analysis_plan,
    lock_time_to_quality_target,
)

root = Path(__file__).parents[1]
plan_path = root / "research" / "paper-1" / "baseline_analysis.json"
matrix_path = root / "research" / "paper-1" / "baseline_matrix.json"


def result(pair, arm, final_loss, created_at):
    steps = [0, 488, 976, 1464, 1952, 2000]
    seconds = [0.0, 10.0, 20.0, 30.0, 40.0, 45.0]
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
                "validation_source_losses": {
                    "source-a": loss,
                    "source-b": loss - 0.1,
                },
                "validation_tokens": 20_000_000 if step == 2000 else 5_000_000,
                "optimizer_seconds": elapsed + 3.0,
                "steady_training_seconds": elapsed,
            }
        )
    geometry = {
        "dense_global_param_match": (153_977_088, 1_301_237_760),
        "five_cache_kda_gqa": (153_958_938, 1_021_601_280),
    }[arm]
    return {
        "format": "speck_paper_baseline_run_result",
        "format_version": 1,
        "status": "complete_qualified",
        "created_at": created_at,
        "paper_id": "speck-paper-1",
        "policy_id": "architecture-promotion-v1",
        "analysis_plan_sha256": file_sha256(plan_path),
        "baseline_matrix_sha256": file_sha256(matrix_path),
        "pair": pair,
        "arm_id": arm,
        "parameters": geometry[0],
        "flops_per_token_at_4096": geometry[1],
        "training_tokens": 131_072_000,
        "validation_history": history,
        "final_validation": history[-1],
        "non_finite_steps": 0,
    }


def write_results(tmp_path, candidate_delta=-0.02):
    _, plan, _, _ = load_analysis_plan(plan_path)
    controls = []
    candidates = []
    control_losses = [3.0, 3.01, 2.99]
    for pair, control_loss in zip(plan["pairs"], control_losses):
        control_path = tmp_path / f"control-{pair['pair']}.json"
        atomic_json(
            control_path,
            result(pair, "dense_global_param_match", control_loss, "2026-09-05T00:00:00+00:00"),
        )
        controls.append(control_path)
        candidate_path = tmp_path / f"candidate-{pair['pair']}.json"
        atomic_json(
            candidate_path,
            result(
                pair,
                "five_cache_kda_gqa",
                control_loss + candidate_delta,
                "9999-09-05T00:00:00+00:00",
            ),
        )
        candidates.append(candidate_path)
    return controls, candidates


def test_frozen_baseline_analysis_passes_a_clear_noninferior_result(tmp_path):
    controls, candidates = write_results(tmp_path)
    target = lock_time_to_quality_target(plan_path, controls)
    target_path = tmp_path / "target.json"
    atomic_json(target_path, target)

    report = analyze_baselines(plan_path, target_path, controls + candidates)

    assert target["validation_loss_target"] == 3.01
    assert report["status"] == "complete_proxy_evidence_no_promotion_authority"
    assert report["fixed_tokens"]["mean"] == pytest.approx(-0.02)
    assert report["fixed_tokens"]["non_inferiority_pass"]
    assert report["proxy_quality_screen_pass"]
    assert not report["time_to_quality"]["right_censored_pairs"]
    assert report["time_to_quality"]["paired_relative_improvement"]["mean"] > 0


def test_baseline_analysis_retains_right_censoring(tmp_path):
    controls, candidates = write_results(tmp_path, candidate_delta=0.2)
    target_path = tmp_path / "target.json"
    atomic_json(target_path, lock_time_to_quality_target(plan_path, controls))

    report = analyze_baselines(plan_path, target_path, controls + candidates)

    assert report["time_to_quality"]["right_censored_pairs"] == [0, 1, 2]
    assert report["time_to_quality"]["paired_relative_improvement"] is None
    assert not report["fixed_tokens"]["non_inferiority_pass"]
    assert not report["proxy_quality_screen_pass"]


def test_target_lock_rejects_candidate_contamination(tmp_path):
    controls, candidates = write_results(tmp_path)

    with pytest.raises(ValueError, match="three control"):
        lock_time_to_quality_target(plan_path, controls[:2] + candidates[:1])


def test_analysis_requires_every_frozen_cell_once(tmp_path):
    controls, candidates = write_results(tmp_path)
    target_path = tmp_path / "target.json"
    atomic_json(target_path, lock_time_to_quality_target(plan_path, controls))

    with pytest.raises(ValueError, match="every frozen cell"):
        analyze_baselines(plan_path, target_path, controls + candidates[:-1] + candidates[:1])


def test_analysis_plan_rejects_matrix_pin_drift(tmp_path):
    copied = tmp_path / "baseline_analysis.json"
    value = deepcopy(json.loads(plan_path.read_text(encoding="utf-8")))
    value["baseline_matrix_sha256"] = "0" * 64
    copied.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="matrix"):
        load_analysis_plan(copied)


def test_collect_run_result_qualifies_the_frozen_checkpoint_contract(tmp_path):
    pair = {"pair": 0, "seed": 42, "data_token_offset": 0}
    report = result(
        pair,
        "dense_global_param_match",
        3.0,
        "2026-09-05T00:00:00+00:00",
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model_002000.pt").write_bytes(b"model")
    (checkpoint / "optimizer_002000.pt").write_bytes(b"optimizer")
    metadata = {
        "step": 2000,
        "global_tokens": 131_072_000,
        "partial": False,
        "manifest": "b84b09e0b701e35d84487cf6f91e6da9c9fb686b7f6efe67b2e2f5f301fda98e",
        "validation_step": 2000,
        "validation_tokens": 20_000_000,
        "validation_history": report["validation_history"],
        "peak_allocated_bytes": 123,
        "resolved": {
            "seed": 42,
            "data_token_offset": 0,
            "train_tokens": 131_072_000,
            "batch_tokens": 65_536,
            "sequence_length": 4_096,
            "parameters": 153_977_088,
        },
    }
    (checkpoint / "metadata_002000.json").write_text(json.dumps(metadata), encoding="utf-8")
    timing = {
        "optimizer_seconds": 48.0,
        "steady_training_seconds": 45.0,
        "startup_optimizer_seconds": 3.0,
    }
    (checkpoint / "timing_002000.json").write_text(json.dumps(timing), encoding="utf-8")
    (checkpoint / "complete_002000").write_text("complete\n", encoding="utf-8")
    summary = {
        "partial": False,
        "completed_steps": 2000,
        "global_tokens": 131_072_000,
        "validation_history": report["validation_history"],
    }
    (checkpoint / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    experiment = (
        root
        / "experiments"
        / "Speck-Paper1-Baselines-131M"
        / "runs"
        / "pair-0-seed-42-order-0"
        / "dense_global_param_match"
    )

    collected = collect_run_result(plan_path, experiment, checkpoint)

    assert collected["status"] == "complete_qualified"
    assert collected["pair"] == pair
    assert collected["arm_id"] == "dense_global_param_match"
    assert collected["checkpoint"]["model_sha256"] == file_sha256(checkpoint / "model_002000.pt")
    assert collected["peak_allocated_bytes"] == 123
