import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from speck.paper_finalist_systems_telemetry import (
    GPU_UUID,
    atomic_json,
    file_sha256,
    integrate_trace,
    integrate_trial,
    integrate_window,
    validate_samples,
)

root = Path(__file__).parents[1]
protocol_path = root / "research" / "paper-1" / "finalist_systems_v2.json"
protocol = json.loads(protocol_path.read_text(encoding="utf-8"))


def sample(second, power, *, limit=350):
    return {
        "monotonic_ns": second * 1_000_000_000,
        "utc": (datetime(2026, 9, 6, tzinfo=timezone.utc) + timedelta(seconds=second)).isoformat(),
        "gpu_uuid": GPU_UUID,
        "power_watts": power,
        "temperature_c": 44,
        "graphics_clock_mhz": 1700,
        "memory_clock_mhz": 9500,
        "power_limit_watts": limit,
        "pstate": "P2",
        "gpu_utilization_percent": 90 if power > 100 else 0,
        "memory_utilization_percent": 40 if power > 100 else 0,
        "fan_percent": 50,
        "active_throttle_reasons": "None",
        "cpu_utilization_percent": 10,
        "load_average_1m": 1.0,
        "available_memory_bytes": 20_000_000_000,
        "disk_read_bytes": second * 100,
        "disk_write_bytes": second * 10,
        "display_state": "unchanged",
        "gpu_process_pids": [123] if power > 100 else [],
        "benchmark_pids": [123] if power > 100 else [],
    }


def complete_trace():
    samples = []
    for second in range(281):
        power = 30 if second <= 130 or second >= 150 else 200
        samples.append(sample(second, power))
    intervals = {
        "pre_idle_start_ns": 0,
        "pre_idle_end_ns": 120_000_000_000,
        "measured_start_ns": 130_000_000_000,
        "measured_end_ns": 150_000_000_000,
        "post_idle_start_ns": 160_000_000_000,
        "post_idle_end_ns": 280_000_000_000,
    }
    markers = {
        "process_start": 120_000_000_000,
        "checkpoint_loaded": 122_000_000_000,
        "batches_materialized": 124_000_000_000,
        "warmup_start": 125_000_000_000,
        "warmup_end": 130_000_000_000,
        "measured_start": 130_000_000_000,
        "measured_end": 150_000_000_000,
        "process_end": 160_000_000_000,
    }
    return samples, intervals, markers


def test_complete_trace_integrates_gross_idle_and_incremental_energy():
    samples, intervals, markers = complete_trace()
    result = integrate_trial(samples, intervals, markers, protocol)
    assert result["qualified"] is True
    assert result["gross_board_energy_joules"]["estimate"] == pytest.approx(3830)
    assert result["idle_board_power_watts"]["estimate"] == pytest.approx(30)
    assert result["incremental_board_energy_joules"]["estimate"] == pytest.approx(3230)
    assert result["telemetry"]["measured_window_coverage"] == 1
    assert result["telemetry"]["maximum_sample_gap_seconds"] == 1


def test_missing_sample_uses_zero_to_power_limit_bounds_and_fails_coverage():
    samples, _, _ = complete_trace()
    samples = [value for value in samples if value["monotonic_ns"] != 140_000_000_000]
    measured = integrate_window(samples, 130_000_000_000, 150_000_000_000)
    assert measured["missing_sample_seconds"] == 1
    assert measured["measured_window_coverage"] == pytest.approx(0.95)
    assert measured["qualified"] is False
    assert measured["energy_joules"]["lower"] < measured["energy_joules"]["estimate"]
    assert measured["energy_joules"]["upper"] > measured["energy_joules"]["estimate"]


def test_trace_requires_strict_monotonic_timestamps():
    samples, _, _ = complete_trace()
    samples[2]["monotonic_ns"] = samples[1]["monotonic_ns"]
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_samples(samples)


def test_trace_requires_strict_UTC_timestamps():
    samples, _, _ = complete_trace()
    samples[2]["utc"] = samples[1]["utc"]
    with pytest.raises(ValueError, match="UTC timestamps are not strictly increasing"):
        validate_samples(samples)


def test_trace_rejects_wrong_gpu_uuid():
    samples, _, _ = complete_trace()
    samples[5]["gpu_uuid"] = "GPU-wrong"
    with pytest.raises(ValueError, match="GPU UUID"):
        validate_samples(samples)


def test_trace_rejects_power_above_recorded_limit():
    samples, _, _ = complete_trace()
    samples[15]["power_watts"] = 400
    with pytest.raises(ValueError, match="physical bound"):
        validate_samples(samples)


def test_trace_requires_bracketing_samples():
    samples, _, _ = complete_trace()
    with pytest.raises(ValueError, match="bracket"):
        integrate_window(validate_samples(samples), -1_000_000_000, 5_000_000_000)


def test_trace_file_is_bound_to_frozen_protocol(tmp_path):
    samples, intervals, markers = complete_trace()
    trace = {
        "format": "speck_paper_finalist_systems_telemetry_trace",
        "format_version": 1,
        "protocol_sha256": file_sha256(protocol_path),
        "samples": samples,
        "intervals": intervals,
        "phase_markers": markers,
    }
    path = tmp_path / "trace.json"
    atomic_json(path, trace)
    result = integrate_trace(protocol_path, path)
    assert result["qualified"] is True
    changed = deepcopy(trace)
    changed["protocol_sha256"] = "0" * 64
    atomic_json(path, changed)
    with pytest.raises(ValueError, match="frozen protocol"):
        integrate_trace(protocol_path, path)


def test_trace_requires_every_ordered_phase_marker():
    samples, intervals, markers = complete_trace()
    markers.pop("checkpoint_loaded")
    with pytest.raises(ValueError, match="phase markers"):
        integrate_trial(samples, intervals, markers, protocol)


def test_trace_requires_frozen_idle_duration():
    samples, intervals, markers = complete_trace()
    intervals["pre_idle_end_ns"] -= 1_000_000_000
    with pytest.raises(ValueError, match="idle duration"):
        integrate_trial(samples, intervals, markers, protocol)
