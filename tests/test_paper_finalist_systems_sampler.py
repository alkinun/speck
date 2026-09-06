from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import speck.paper_finalist_systems_sampler as sampler_module
from speck.paper_finalist_systems_sampler import (
    GPU_FIELDS,
    cpu_utilization,
    parse_available_memory,
    parse_diskstats,
    parse_gpu_csv,
    parse_process_csv,
    sample_once,
    sample_series,
    serialize_sample_series,
)
from speck.paper_finalist_systems_telemetry import GPU_UUID, file_sha256

root = Path(__file__).parents[1]
protocol_path = root / "research" / "paper-1" / "finalist_systems_v2.json"


def gpu_row(*, uuid=GPU_UUID, power="210.5"):
    values = {
        "uuid": uuid,
        "power.draw": power,
        "temperature.gpu": "65",
        "clocks.current.graphics": "1725",
        "clocks.current.memory": "9501",
        "power.limit": "350",
        "pstate": "P2",
        "utilization.gpu": "98",
        "utilization.memory": "55",
        "fan.speed": "60",
        "clocks_throttle_reasons.active": "0x0000000000000000",
        "display_active": "Enabled",
    }
    return ", ".join(values[name] for name in GPU_FIELDS) + "\n"


def host_reader(path):
    return {
        "/proc/stat": "cpu  100 0 50 800 50 0 0 0 0 0\n",
        "/proc/loadavg": "1.25 1.00 0.75 1/100 123\n",
        "/proc/meminfo": "MemTotal: 32000000 kB\nMemAvailable: 20000000 kB\n",
        "/proc/diskstats": "259 0 nvme0n1 10 0 100 0 20 0 40 0 0 0 0 0 0 0 0\n",
    }[path]


def runner(command):
    if command[1].startswith("--query-gpu"):
        return gpu_row()
    return f"123, {GPU_UUID}\n456, GPU-other\n"


def test_gpu_csv_parses_exact_frozen_field_vector():
    parsed = parse_gpu_csv(gpu_row())
    assert parsed["gpu_uuid"] == GPU_UUID
    assert parsed["power_watts"] == 210.5
    assert parsed["graphics_clock_mhz"] == 1725
    assert parsed["active_throttle_reasons"] == "0x0000000000000000"
    assert parsed["display_state"] == "Enabled"


def test_gpu_csv_rejects_unsupported_numeric_field():
    with pytest.raises(ValueError, match="invalid power draw"):
        parse_gpu_csv(gpu_row(power="[Not Supported]"))


def test_process_csv_filters_exact_gpu_and_deduplicates():
    output = f"123, {GPU_UUID}\n123, {GPU_UUID}\n456, GPU-other\n"
    assert parse_process_csv(output, GPU_UUID) == [123]


def test_host_counter_parsers_are_exact():
    assert parse_available_memory(host_reader("/proc/meminfo")) == 20_480_000_000
    assert parse_diskstats(host_reader("/proc/diskstats"), {"nvme0n1"}) == {
        "disk_read_bytes": 51_200,
        "disk_write_bytes": 20_480,
    }
    assert (
        cpu_utilization(
            {"total": 1000, "idle": 800},
            {"total": 1100, "idle": 850},
        )
        == 50
    )


def test_diskstats_requires_every_explicit_device():
    with pytest.raises(ValueError, match="missing a selected"):
        parse_diskstats(host_reader("/proc/diskstats"), {"nvme0n1", "sda"})


def test_sample_once_composes_mocked_gpu_host_and_process_data():
    sample, cpu = sample_once(
        None,
        [123],
        runner=runner,
        reader=host_reader,
        device_names={"nvme0n1"},
        monotonic_ns=lambda: 1_000_000_000,
        utc_now=lambda: datetime(2026, 9, 6, tzinfo=timezone.utc),
    )
    assert sample["gpu_uuid"] == GPU_UUID
    assert sample["gpu_process_pids"] == [123]
    assert sample["benchmark_pids"] == [123]
    assert sample["cpu_utilization_percent"] == 0
    assert cpu == {"total": 1000, "idle": 850}


def test_sample_series_uses_one_hz_sequence_and_schema_validation():
    index = 0

    def sampler(previous_cpu, benchmark_pids):
        nonlocal index
        current = index
        index += 1
        sample = {
            "monotonic_ns": current * 1_000_000_000,
            "utc": (
                datetime(2026, 9, 6, tzinfo=timezone.utc) + timedelta(seconds=current)
            ).isoformat(),
            "gpu_uuid": GPU_UUID,
            "power_watts": 30,
            "temperature_c": 44,
            "graphics_clock_mhz": 200,
            "memory_clock_mhz": 405,
            "power_limit_watts": 350,
            "pstate": "P8",
            "gpu_utilization_percent": 0,
            "memory_utilization_percent": 0,
            "fan_percent": 30,
            "active_throttle_reasons": "0x0",
            "cpu_utilization_percent": 0,
            "load_average_1m": 1,
            "available_memory_bytes": 20_000_000_000,
            "disk_read_bytes": current * 512,
            "disk_write_bytes": current * 256,
            "display_state": "Enabled",
            "gpu_process_pids": [],
            "benchmark_pids": benchmark_pids,
        }
        return sample, {"total": current + 1, "idle": current + 1}

    sleeps = []
    clocks = iter((0.0, 0.2, 1.2))
    samples = sample_series(
        3,
        [],
        sampler=sampler,
        sleeper=sleeps.append,
        monotonic=lambda: next(clocks),
    )
    assert [sample["monotonic_ns"] for sample in samples] == [0, 1_000_000_000, 2_000_000_000]
    assert sleeps == pytest.approx([0.8, 0.8])


def test_sample_series_rejects_non_protocol_frequency():
    with pytest.raises(ValueError, match="1 Hz"):
        sample_series(
            2, [], interval_seconds=0.5, sampler=SimpleNamespace(), sleeper=lambda _: None
        )


def test_serialized_series_is_bound_to_protocol_and_explicit_devices(monkeypatch):
    captured = {}

    def fake_series(count, benchmark_pids, **kwargs):
        captured.update(count=count, benchmark_pids=benchmark_pids, **kwargs)
        return [{"sample": 0}, {"sample": 1}]

    monkeypatch.setattr(sampler_module, "sample_series", fake_series)
    result = serialize_sample_series(protocol_path, 2, [123], ["nvme0n1", "nvme0n1"])
    assert result["protocol_sha256"] == file_sha256(protocol_path)
    assert result["disk_devices"] == ["nvme0n1"]
    assert captured["device_names"] == {"nvme0n1"}
