import json
from pathlib import Path

import pytest

from speck.paper_finalist_systems_analysis import analyze_systems, atomic_json, file_sha256

root = Path(__file__).parents[1]
protocol_path = root / "research" / "paper-1" / "finalist_systems_v2.json"
protocol = json.loads(protocol_path.read_text(encoding="utf-8"))


def energy(estimate, spread=0.01):
    return {
        "lower": estimate * (1 - spread),
        "estimate": estimate,
        "upper": estimate * (1 + spread),
    }


def trial(arm, wall, gross, incremental, *, temperature=44):
    return {
        "status": "complete_qualified",
        "arm_id": protocol["arms"][arm],
        "measured_optimizer_steps": 30,
        "measured_tokens": 1_966_080,
        "measured_wall_seconds": wall,
        "gross_board_energy_joules": energy(gross),
        "incremental_board_energy_joules": energy(incremental),
        "start_temperature_c": temperature,
        "telemetry": {
            "measured_window_coverage": 1.0,
            "maximum_sample_gap_seconds": 1.0,
        },
        "peak_allocated_bytes": 10_000,
        "peak_reserved_bytes": 12_000,
        "peak_nvml_used_bytes": 14_000,
        "non_finite_steps": 0,
        "OOM": False,
        "kernel_fallback": False,
    }


def write_blocks(tmp_path, *, candidate_time=8, candidate_energy=800):
    paths = []
    protocol_sha = file_sha256(protocol_path)
    for expected in protocol["paired_blocks"]:
        report = {
            "format": "speck_paper_finalist_systems_block_result",
            "format_version": 1,
            "status": "complete_qualified",
            "protocol_sha256": protocol_sha,
            "block": expected["block"],
            "pair": expected["pair"],
            "trial_order": expected["trial_order"],
            "trials": {
                "control": trial("control", 10, 1000, 800),
                "candidate": trial("candidate", candidate_time, candidate_energy, 600),
            },
        }
        path = tmp_path / f"block-{expected['block']}.json"
        atomic_json(path, report)
        paths.append(path)
    return paths


def test_systems_analysis_passes_clear_time_and_energy_improvement(tmp_path):
    report = analyze_systems(protocol_path, write_blocks(tmp_path))
    assert report["status"] == "complete_joint_training_systems_efficiency_pass"
    assert report["joint_training_systems_efficiency_pass"] is True
    assert report["primary"]["measured_wall_seconds_per_token"][
        "geometric_mean_ratio"
    ] == pytest.approx(0.8)
    assert report["primary"]["measured_wall_seconds_per_token"]["exact_sign_flip_p"] == 1 / 64
    assert report["primary"]["gross_board_joules_per_token_conservative"]["pass"] is True


def test_systems_analysis_does_not_turn_time_only_win_into_joint_claim(tmp_path):
    report = analyze_systems(protocol_path, write_blocks(tmp_path, candidate_energy=1200))
    assert report["primary"]["measured_wall_seconds_per_token"]["pass"] is True
    assert report["primary"]["gross_board_joules_per_token_conservative"]["pass"] is False
    assert report["joint_training_systems_efficiency_pass"] is False
    assert report["claim_boundary"]["same_RTX_3090_4K_BF16_training_step_time"] is True
    assert report["claim_boundary"]["gross_GPU_board_energy"] is False


def test_systems_analysis_rejects_order_stratum_reversal(tmp_path):
    paths = write_blocks(tmp_path)
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value["trial_order"][0] == "control":
            value["trials"]["candidate"]["measured_wall_seconds"] = 11
        else:
            value["trials"]["candidate"]["measured_wall_seconds"] = 6
        atomic_json(path, value)
    report = analyze_systems(protocol_path, paths)
    timing = report["primary"]["measured_wall_seconds_per_token"]
    assert timing["order_mean_log_ratio"]["control_then_candidate"] > 0
    assert timing["pass"] is False
    assert report["joint_training_systems_efficiency_pass"] is False


def test_systems_analysis_fails_claim_with_missing_block(tmp_path):
    paths = write_blocks(tmp_path)
    report = analyze_systems(protocol_path, paths[:-1])
    assert report["status"] == "incomplete_failed_no_training_systems_claim"
    assert report["missing_blocks"] == [5]
    assert report["primary"]["measured_wall_seconds_per_token"]["n"] == 5
    assert report["joint_training_systems_efficiency_pass"] is False


def test_systems_analysis_uses_conservative_energy_bounds(tmp_path):
    paths = write_blocks(tmp_path)
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        value["trials"]["control"]["gross_board_energy_joules"] = {
            "lower": 700,
            "estimate": 1000,
            "upper": 1100,
        }
        value["trials"]["candidate"]["gross_board_energy_joules"] = {
            "lower": 500,
            "estimate": 800,
            "upper": 900,
        }
        atomic_json(path, value)
    report = analyze_systems(protocol_path, paths)
    assert report["primary"]["gross_board_joules_per_token_point"]["pass"] is True
    assert report["primary"]["gross_board_joules_per_token_conservative"]["pass"] is False
    assert report["joint_training_systems_efficiency_pass"] is False


def test_systems_analysis_fails_energy_claim_when_control_lower_bound_is_zero(tmp_path):
    paths = write_blocks(tmp_path)
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        value["trials"]["control"]["gross_board_energy_joules"]["lower"] = 0
        atomic_json(path, value)
    report = analyze_systems(protocol_path, paths)
    assert report["primary"]["gross_board_joules_per_token_conservative"]["n"] == 0
    assert report["primary"]["gross_board_joules_per_token_conservative"]["pass"] is False
    assert report["joint_training_systems_efficiency_pass"] is False


def test_systems_analysis_retains_failed_block(tmp_path):
    paths = write_blocks(tmp_path)
    value = json.loads(paths[2].read_text(encoding="utf-8"))
    value["status"] = "failed_retained"
    value["failure"] = "telemetry coverage below 0.99"
    value.pop("trials")
    atomic_json(paths[2], value)
    report = analyze_systems(protocol_path, paths)
    assert report["failed_blocks"] == [2]
    assert report["valid_blocks"] == [0, 1, 3, 4, 5]
    assert report["joint_training_systems_efficiency_pass"] is False


def test_systems_analysis_rejects_temperature_mismatch(tmp_path):
    paths = write_blocks(tmp_path)
    value = json.loads(paths[0].read_text(encoding="utf-8"))
    value["trials"]["candidate"]["start_temperature_c"] = 41
    atomic_json(paths[0], value)
    with pytest.raises(ValueError, match="start-temperature"):
        analyze_systems(protocol_path, paths)
