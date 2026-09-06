"""Validate finalist systems telemetry and integrate conservative board-energy intervals."""

import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path

GPU_UUID = "GPU-6e7f2d05-ac19-3812-ff41-33079fd67bfd"
NUMERIC_FIELDS = (
    "power_watts",
    "temperature_c",
    "graphics_clock_mhz",
    "memory_clock_mhz",
    "power_limit_watts",
    "gpu_utilization_percent",
    "memory_utilization_percent",
    "fan_percent",
)
REQUIRED_FIELDS = {
    "monotonic_ns",
    "utc",
    "gpu_uuid",
    *NUMERIC_FIELDS,
    "pstate",
    "active_throttle_reasons",
    "cpu_utilization_percent",
    "load_average_1m",
    "available_memory_bytes",
    "disk_read_bytes",
    "disk_write_bytes",
    "display_state",
    "gpu_process_pids",
    "benchmark_pids",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path):
    path = Path(path).expanduser().resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return path, value


def atomic_json(path, value):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _number(value, name, *, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"telemetry sample has invalid {name}")
    value = float(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"telemetry sample has out-of-range {name}")
    if maximum is not None and value > maximum:
        raise ValueError(f"telemetry sample has out-of-range {name}")
    return value


def validate_samples(samples, gpu_uuid=GPU_UUID):
    if not isinstance(samples, list) or len(samples) < 2:
        raise ValueError("telemetry requires at least two samples")
    validated = []
    previous_ns = -1
    previous_utc = None
    for sample in samples:
        if not isinstance(sample, dict) or not REQUIRED_FIELDS <= set(sample):
            raise ValueError("telemetry sample is incomplete")
        monotonic_ns = sample["monotonic_ns"]
        if (
            isinstance(monotonic_ns, bool)
            or not isinstance(monotonic_ns, int)
            or monotonic_ns <= previous_ns
        ):
            raise ValueError("telemetry monotonic timestamps are not strictly increasing")
        previous_ns = monotonic_ns
        try:
            utc = datetime.fromisoformat(sample["utc"])
        except (TypeError, ValueError) as error:
            raise ValueError("telemetry UTC timestamp is invalid") from error
        if utc.tzinfo is None:
            raise ValueError("telemetry UTC timestamp is not timezone-aware")
        if previous_utc is not None and utc <= previous_utc:
            raise ValueError("telemetry UTC timestamps are not strictly increasing")
        previous_utc = utc
        if sample["gpu_uuid"] != gpu_uuid:
            raise ValueError("telemetry GPU UUID changed")
        numeric = {name: _number(sample[name], name, minimum=0) for name in NUMERIC_FIELDS}
        if (
            numeric["power_watts"] > numeric["power_limit_watts"]
            or numeric["temperature_c"] > 120
            or numeric["gpu_utilization_percent"] > 100
            or numeric["memory_utilization_percent"] > 100
            or numeric["fan_percent"] > 100
        ):
            raise ValueError("telemetry GPU field is outside its physical bound")
        for name in ("cpu_utilization_percent", "load_average_1m"):
            _number(sample[name], name, minimum=0)
        _number(sample["cpu_utilization_percent"], "cpu utilization", maximum=100)
        for name in ("available_memory_bytes", "disk_read_bytes", "disk_write_bytes"):
            if (
                isinstance(sample[name], bool)
                or not isinstance(sample[name], int)
                or sample[name] < 0
            ):
                raise ValueError(f"telemetry sample has invalid {name}")
        if (
            not isinstance(sample["pstate"], str)
            or not isinstance(sample["active_throttle_reasons"], str)
            or not isinstance(sample["display_state"], str)
            or any(
                not isinstance(values, list)
                or any(
                    isinstance(pid, bool) or not isinstance(pid, int) or pid < 1 for pid in values
                )
                for values in (sample["gpu_process_pids"], sample["benchmark_pids"])
            )
        ):
            raise ValueError("telemetry categorical or process field is invalid")
        validated.append(
            {
                **sample,
                **numeric,
            }
        )
    return validated


def _interpolate(left, right, target_ns):
    if target_ns == left["monotonic_ns"]:
        return left
    if target_ns == right["monotonic_ns"]:
        return right
    fraction = (target_ns - left["monotonic_ns"]) / (right["monotonic_ns"] - left["monotonic_ns"])
    return {
        "monotonic_ns": target_ns,
        "power_watts": left["power_watts"]
        + fraction * (right["power_watts"] - left["power_watts"]),
        "power_limit_watts": max(left["power_limit_watts"], right["power_limit_watts"]),
    }


def _boundary(samples, target_ns):
    for sample in samples:
        if sample["monotonic_ns"] == target_ns:
            return sample
    before = [sample for sample in samples if sample["monotonic_ns"] < target_ns]
    after = [sample for sample in samples if sample["monotonic_ns"] > target_ns]
    if not before or not after:
        raise ValueError("telemetry does not bracket an interval boundary")
    return _interpolate(before[-1], after[0], target_ns)


def integrate_window(samples, start_ns, end_ns, sampling_hz=1):
    if (
        isinstance(start_ns, bool)
        or not isinstance(start_ns, int)
        or isinstance(end_ns, bool)
        or not isinstance(end_ns, int)
        or start_ns >= end_ns
        or sampling_hz != 1
    ):
        raise ValueError("telemetry interval or sampling frequency is invalid")
    start = _boundary(samples, start_ns)
    end = _boundary(samples, end_ns)
    interior = [sample for sample in samples if start_ns < sample["monotonic_ns"] < end_ns]
    points = [start, *interior, end]
    nominal_seconds = 1 / sampling_hz
    point_energy = 0.0
    lower_energy = 0.0
    upper_energy = 0.0
    missing_seconds = 0.0
    maximum_gap = 0.0
    for left, right in zip(points, points[1:]):
        duration = (right["monotonic_ns"] - left["monotonic_ns"]) / 1e9
        maximum_gap = max(maximum_gap, duration)
        missing = max(0.0, duration - nominal_seconds)
        observed = duration - missing
        average_power = (left["power_watts"] + right["power_watts"]) / 2
        point_energy += duration * average_power
        lower_energy += observed * average_power
        upper_energy += observed * average_power + missing * max(
            left["power_limit_watts"], right["power_limit_watts"]
        )
        missing_seconds += missing
    elapsed = (end_ns - start_ns) / 1e9
    coverage = max(0.0, 1 - missing_seconds / elapsed)
    return {
        "start_monotonic_ns": start_ns,
        "end_monotonic_ns": end_ns,
        "elapsed_seconds": elapsed,
        "samples_including_boundaries": len(points),
        "maximum_sample_gap_seconds": maximum_gap,
        "missing_sample_seconds": missing_seconds,
        "measured_window_coverage": coverage,
        "energy_joules": {
            "lower": lower_energy,
            "estimate": point_energy,
            "upper": upper_energy,
        },
        "qualified": coverage >= 0.99 and maximum_gap <= 2.5,
    }


def _validate_phase_markers(markers, intervals, protocol):
    expected = protocol["telemetry"]["trial_phase_markers"]
    if not isinstance(markers, dict) or set(markers) != set(expected):
        raise ValueError("telemetry trial phase markers are incomplete or out of order")
    values = [markers[name] for name in expected]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values) or any(
        right < left for left, right in zip(values, values[1:])
    ):
        raise ValueError("telemetry trial phase timestamps are invalid")
    if (
        markers["warmup_end"] != markers["measured_start"]
        or markers["measured_start"] != intervals["measured_start_ns"]
        or markers["measured_end"] != intervals["measured_end_ns"]
        or intervals["pre_idle_end_ns"] > markers["process_start"]
        or markers["process_end"] > intervals["post_idle_start_ns"]
    ):
        raise ValueError("telemetry trial phase markers do not match interval boundaries")


def integrate_trial(samples, intervals, phase_markers, protocol):
    samples = validate_samples(samples, protocol["hardware_and_software_lock"]["gpu_uuid"])
    required = {
        "pre_idle_start_ns",
        "pre_idle_end_ns",
        "measured_start_ns",
        "measured_end_ns",
        "post_idle_start_ns",
        "post_idle_end_ns",
    }
    if not isinstance(intervals, dict) or set(intervals) != required:
        raise ValueError("telemetry trial intervals are incomplete")
    thermal = protocol["thermal_and_idle_design"]
    if (
        intervals["pre_idle_end_ns"] - intervals["pre_idle_start_ns"]
        != thermal["pre_block_idle_seconds"] * 1_000_000_000
        or intervals["post_idle_end_ns"] - intervals["post_idle_start_ns"]
        != thermal["post_block_idle_seconds"] * 1_000_000_000
    ):
        raise ValueError("telemetry flanking idle duration changed")
    _validate_phase_markers(phase_markers, intervals, protocol)
    telemetry = protocol["telemetry"]
    pre = integrate_window(
        samples,
        intervals["pre_idle_start_ns"],
        intervals["pre_idle_end_ns"],
        telemetry["sampling_hz"],
    )
    measured = integrate_window(
        samples,
        intervals["measured_start_ns"],
        intervals["measured_end_ns"],
        telemetry["sampling_hz"],
    )
    post = integrate_window(
        samples,
        intervals["post_idle_start_ns"],
        intervals["post_idle_end_ns"],
        telemetry["sampling_hz"],
    )
    idle_seconds = pre["elapsed_seconds"] + post["elapsed_seconds"]
    idle = {
        key: (pre["energy_joules"][key] + post["energy_joules"][key]) / idle_seconds
        for key in ("lower", "estimate", "upper")
    }
    elapsed = measured["elapsed_seconds"]
    gross = measured["energy_joules"]
    incremental = {
        "lower": gross["lower"] - elapsed * idle["upper"],
        "estimate": gross["estimate"] - elapsed * idle["estimate"],
        "upper": gross["upper"] - elapsed * idle["lower"],
    }
    return {
        "gross_board_energy_joules": gross,
        "idle_board_power_watts": idle,
        "incremental_board_energy_joules": incremental,
        "telemetry": {
            "measured_window_coverage": measured["measured_window_coverage"],
            "maximum_sample_gap_seconds": measured["maximum_sample_gap_seconds"],
            "measured_missing_sample_seconds": measured["missing_sample_seconds"],
            "pre_idle_qualified": pre["qualified"],
            "measured_qualified": measured["qualified"],
            "post_idle_qualified": post["qualified"],
        },
        "qualified": pre["qualified"] and measured["qualified"] and post["qualified"],
    }


def integrate_trace(protocol_path, trace_path):
    protocol_path, protocol = load_object(protocol_path)
    trace_path, trace = load_object(trace_path)
    protocol_sha = file_sha256(protocol_path)
    if (
        protocol.get("format") != "speck_paper_finalist_systems_protocol"
        or protocol.get("format_version") != 2
        or trace.get("format") != "speck_paper_finalist_systems_telemetry_trace"
        or trace.get("format_version") != 1
        or trace.get("protocol_sha256") != protocol_sha
    ):
        raise ValueError("systems telemetry trace does not match the frozen protocol")
    return {
        "format": "speck_paper_finalist_systems_energy_integration",
        "format_version": 1,
        "protocol": {"path": str(protocol_path), "sha256": protocol_sha},
        "trace": {"path": str(trace_path), "sha256": file_sha256(trace_path)},
        **integrate_trial(
            trace.get("samples"),
            trace.get("intervals"),
            trace.get("phase_markers"),
            protocol,
        ),
    }
