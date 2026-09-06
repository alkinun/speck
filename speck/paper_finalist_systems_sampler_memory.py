"""Add same-row NVML used-memory telemetry to the frozen finalist systems sampler."""

import csv
import io
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from speck.paper_finalist_systems_sampler import (
    GPU_FIELDS,
    GPU_UUID,
    _number,
    _run,
    cpu_utilization,
    file_sha256,
    parse_available_memory,
    parse_cpu_stat,
    parse_diskstats,
    parse_gpu_csv,
    parse_process_csv,
    sample_series,
)
from speck.paper_finalist_systems_telemetry import validate_samples

GPU_FIELDS_WITH_MEMORY = (GPU_FIELDS[0], "memory.used", *GPU_FIELDS[1:])


def load_memory_contract(path):
    path = Path(path).expanduser().resolve()
    contract = json.loads(path.read_text(encoding="utf-8"))
    root = path.parents[2]
    if (
        contract.get("format") != "speck_paper_finalist_systems_memory_supplement"
        or contract.get("format_version") != 1
        or contract.get("status")
        != "frozen_pre_result_additive_secondary_measurement_execution_blocked"
    ):
        raise ValueError("invalid finalist systems memory supplement")
    for reference in contract.get("inputs", {}).values():
        source = root / reference.get("path", "")
        if not source.is_file() or file_sha256(source) != reference.get("sha256"):
            raise ValueError("finalist systems memory supplement input changed")
    query = contract.get("additive_query", {})
    reduction = contract.get("reduction", {})
    decision = contract.get("decision", {})
    if (
        query.get("field") != "memory.used"
        or query.get("same_GPU_CSV_row_as_power_and_clocks") is not True
        or query.get("reported_unit_under_nounits") != "MiB"
        or query.get("bytes_per_MiB") != 1_048_576
        or query.get("serialized_field") != "memory_used_bytes"
        or reduction.get("statistic") != "maximum memory_used_bytes"
        or reduction.get("serialized_trial_field") != "peak_nvml_used_bytes"
        or decision.get("v1_sampler_modified") is not False
        or decision.get("v2_protocol_modified") is not False
        or decision.get("execution_authorized") is not False
    ):
        raise ValueError("finalist systems memory supplement semantics changed")
    return path, contract


def parse_gpu_csv_with_memory(output):
    rows = list(csv.reader(io.StringIO(output), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != len(GPU_FIELDS_WITH_MEMORY):
        raise ValueError("nvidia-smi must return exactly one complete memory-supplemented GPU row")
    values = dict(zip(GPU_FIELDS_WITH_MEMORY, (value.strip() for value in rows[0])))
    memory_mib = _number(values.pop("memory.used"), "used memory")
    if not math.isfinite(memory_mib) or memory_mib < 0:
        raise ValueError("nvidia-smi returned invalid used memory")
    base_row = ",".join(values[field] for field in GPU_FIELDS)
    return {
        **parse_gpu_csv(base_row),
        "memory_used_bytes": round(memory_mib * 1_048_576),
    }


def validate_samples_with_memory(samples):
    samples = validate_samples(samples, GPU_UUID)
    for sample in samples:
        value = sample.get("memory_used_bytes")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("telemetry sample has invalid memory_used_bytes")
    return samples


def peak_memory_used(samples, start_ns, end_ns):
    samples = validate_samples_with_memory(samples)
    if (
        isinstance(start_ns, bool)
        or not isinstance(start_ns, int)
        or isinstance(end_ns, bool)
        or not isinstance(end_ns, int)
        or start_ns >= end_ns
    ):
        raise ValueError("NVML peak interval is invalid")
    selected = [sample for sample in samples if start_ns <= sample["monotonic_ns"] <= end_ns]
    if (
        len(selected) < 2
        or not any(sample["monotonic_ns"] <= start_ns for sample in samples)
        or not any(sample["monotonic_ns"] >= end_ns for sample in samples)
    ):
        raise ValueError("NVML peak interval is missing samples or boundary brackets")
    return max(sample["memory_used_bytes"] for sample in selected)


def sample_once_with_memory(
    previous_cpu,
    benchmark_pids,
    *,
    runner=_run,
    reader=None,
    device_names=None,
    monotonic_ns=time.monotonic_ns,
    utc_now=lambda: datetime.now(timezone.utc),
):
    reader = reader or (lambda path: Path(path).read_text(encoding="utf-8"))
    if not isinstance(device_names, set) or not device_names:
        raise ValueError("telemetry requires explicit backing block-device names")
    gpu = parse_gpu_csv_with_memory(
        runner(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(GPU_FIELDS_WITH_MEMORY)}",
                "--format=csv,noheader,nounits",
            ]
        )
    )
    processes = parse_process_csv(
        runner(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid",
                "--format=csv,noheader,nounits",
            ]
        ),
        gpu["gpu_uuid"],
    )
    cpu = parse_cpu_stat(reader("/proc/stat"))
    load_raw = reader("/proc/loadavg").split()
    if not load_raw:
        raise ValueError("/proc/loadavg is empty")
    return (
        {
            "monotonic_ns": monotonic_ns(),
            "utc": utc_now().isoformat(),
            **gpu,
            "cpu_utilization_percent": cpu_utilization(previous_cpu, cpu),
            "load_average_1m": _number(load_raw[0], "load average"),
            "available_memory_bytes": parse_available_memory(reader("/proc/meminfo")),
            **parse_diskstats(reader("/proc/diskstats"), device_names),
            "gpu_process_pids": processes,
            "benchmark_pids": sorted(set(benchmark_pids)),
        },
        cpu,
    )


def serialize_memory_sample_series(
    protocol_path,
    memory_contract_path,
    count,
    benchmark_pids,
    device_names,
    **sample_kwargs,
):
    protocol_path = Path(protocol_path).expanduser().resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    memory_contract_path, _ = load_memory_contract(memory_contract_path)
    if (
        protocol.get("format") != "speck_paper_finalist_systems_protocol"
        or protocol.get("format_version") != 2
        or protocol.get("telemetry", {}).get("sampling_hz") != 1
    ):
        raise ValueError("memory sampler requires the frozen systems protocol")
    devices = set(device_names)
    if not devices or any(not isinstance(name, str) or not name for name in devices):
        raise ValueError("memory sampler requires explicit valid disk devices")
    samples = sample_series(
        count,
        benchmark_pids,
        sampler=sample_once_with_memory,
        device_names=devices,
        **sample_kwargs,
    )
    samples = validate_samples_with_memory(samples)
    return {
        "format": "speck_paper_finalist_systems_telemetry_samples",
        "format_version": 2,
        "protocol_sha256": file_sha256(protocol_path),
        "memory_supplement_sha256": file_sha256(memory_contract_path),
        "sampling_hz": 1,
        "disk_devices": sorted(devices),
        "samples": samples,
    }
