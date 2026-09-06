from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import speck.paper_finalist_systems_sampler_memory as memory_module
from speck.paper_finalist_systems_sampler import GPU_FIELDS
from speck.paper_finalist_systems_sampler_memory import (
    GPU_FIELDS_WITH_MEMORY,
    load_memory_contract,
    parse_gpu_csv_with_memory,
    peak_memory_used,
    serialize_memory_sample_series,
    validate_samples_with_memory,
)
from speck.paper_finalist_systems_telemetry import GPU_UUID, file_sha256
from tests.test_paper_finalist_systems_sampler import gpu_row
from tests.test_paper_finalist_systems_telemetry import sample

root = Path(__file__).parents[1]
protocol_path = root / "research" / "paper-1" / "finalist_systems_v2.json"
memory_path = root / "research" / "paper-1" / "finalist_systems_memory_v1.json"


def memory_row(memory="12288"):
    base = gpu_row().strip().split(", ")
    values = dict(zip(GPU_FIELDS, base))
    values["memory.used"] = memory
    return ", ".join(values[field] for field in GPU_FIELDS_WITH_MEMORY) + "\n"


def memory_samples():
    values = []
    for second, memory_mib in enumerate((1000, 1200, 1100)):
        value = sample(second, 200)
        value["memory_used_bytes"] = memory_mib * 1_048_576
        values.append(value)
    return values


def test_memory_supplement_is_frozen_and_additive():
    _, contract = load_memory_contract(memory_path)
    assert contract["additive_query"]["field"] == "memory.used"
    assert contract["unchanged"]["primary_endpoints"] == [
        "measured wall seconds per token",
        "gross GPU-board joules per token",
    ]
    assert contract["decision"]["v1_sampler_modified"] is False
    assert contract["decision"]["v2_protocol_modified"] is False


def test_memory_same_row_parser_converts_binary_MiB_to_bytes():
    parsed = parse_gpu_csv_with_memory(memory_row())
    assert parsed["gpu_uuid"] == GPU_UUID
    assert parsed["memory_used_bytes"] == 12_288 * 1_048_576
    assert parsed["power_watts"] == 210.5


def test_memory_same_row_parser_rejects_unsupported_value():
    with pytest.raises(ValueError, match="invalid used memory"):
        parse_gpu_csv_with_memory(memory_row("[Not Supported]"))


def test_memory_schema_rejects_missing_or_noninteger_bytes():
    values = memory_samples()
    values[1].pop("memory_used_bytes")
    with pytest.raises(ValueError, match="memory_used_bytes"):
        validate_samples_with_memory(values)


def test_measured_window_peak_requires_samples_and_boundary_brackets():
    values = memory_samples()
    assert peak_memory_used(values, 0, 2_000_000_000) == 1200 * 1_048_576
    assert peak_memory_used(values, 500_000_000, 2_000_000_000) == 1200 * 1_048_576
    with pytest.raises(ValueError, match="boundary brackets"):
        peak_memory_used(values, -500_000_000, 2_000_000_000)


def test_serialized_memory_series_binds_both_contracts(monkeypatch):
    values = memory_samples()

    def fake_series(count, benchmark_pids, **kwargs):
        assert count == 3
        assert kwargs["sampler"] is memory_module.sample_once_with_memory
        assert kwargs["device_names"] == {"nvme0n1"}
        return values

    monkeypatch.setattr(memory_module, "sample_series", fake_series)
    result = serialize_memory_sample_series(
        protocol_path,
        memory_path,
        3,
        [123],
        ["nvme0n1"],
    )
    assert result["format_version"] == 2
    assert result["protocol_sha256"] == file_sha256(protocol_path)
    assert result["memory_supplement_sha256"] == file_sha256(memory_path)
    assert result["samples"] == values


def test_memory_sample_UTC_fixture_is_strictly_ordered():
    values = memory_samples()
    start = datetime(2026, 9, 6, tzinfo=timezone.utc)
    for index, value in enumerate(values):
        value["utc"] = (start + timedelta(seconds=index)).isoformat()
    assert validate_samples_with_memory(values) == values
