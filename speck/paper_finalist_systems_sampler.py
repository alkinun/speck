"""Acquire protocol-bound finalist systems telemetry samples."""

import csv
import io
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from speck.paper_finalist_systems_telemetry import GPU_UUID, file_sha256, validate_samples

GPU_FIELDS = (
    "uuid",
    "power.draw",
    "temperature.gpu",
    "clocks.current.graphics",
    "clocks.current.memory",
    "power.limit",
    "pstate",
    "utilization.gpu",
    "utilization.memory",
    "fan.speed",
    "clocks_throttle_reasons.active",
    "display_active",
)


def _number(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"nvidia-smi returned invalid {name}: {value}") from error
    return number


def parse_gpu_csv(output):
    rows = list(csv.reader(io.StringIO(output), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != len(GPU_FIELDS):
        raise ValueError("nvidia-smi must return exactly one complete GPU row")
    values = dict(zip(GPU_FIELDS, (value.strip() for value in rows[0])))
    return {
        "gpu_uuid": values["uuid"],
        "power_watts": _number(values["power.draw"], "power draw"),
        "temperature_c": _number(values["temperature.gpu"], "temperature"),
        "graphics_clock_mhz": _number(values["clocks.current.graphics"], "graphics clock"),
        "memory_clock_mhz": _number(values["clocks.current.memory"], "memory clock"),
        "power_limit_watts": _number(values["power.limit"], "power limit"),
        "pstate": values["pstate"],
        "gpu_utilization_percent": _number(values["utilization.gpu"], "GPU utilization"),
        "memory_utilization_percent": _number(values["utilization.memory"], "memory utilization"),
        "fan_percent": _number(values["fan.speed"], "fan speed"),
        "active_throttle_reasons": values["clocks_throttle_reasons.active"],
        "display_state": values["display_active"],
    }


def parse_process_csv(output, gpu_uuid):
    processes = []
    for row in csv.reader(io.StringIO(output), skipinitialspace=True):
        if not row or not any(value.strip() for value in row):
            continue
        if len(row) != 2:
            raise ValueError("nvidia-smi returned an incomplete compute-process row")
        raw_pid, raw_uuid = (value.strip() for value in row)
        if raw_uuid != gpu_uuid:
            continue
        if not raw_pid.isdigit() or int(raw_pid) < 1:
            raise ValueError("nvidia-smi returned an invalid compute-process PID")
        processes.append(int(raw_pid))
    return sorted(set(processes))


def parse_cpu_stat(text):
    first = text.splitlines()[0].split()
    if not first or first[0] != "cpu" or len(first) < 6:
        raise ValueError("/proc/stat has no aggregate CPU row")
    try:
        values = [int(value) for value in first[1:]]
    except ValueError as error:
        raise ValueError("/proc/stat aggregate CPU row is invalid") from error
    idle = values[3] + values[4]
    return {"total": sum(values), "idle": idle}


def cpu_utilization(previous, current):
    if previous is None:
        return 0.0
    total = current["total"] - previous["total"]
    idle = current["idle"] - previous["idle"]
    if total <= 0 or idle < 0 or idle > total:
        raise ValueError("aggregate CPU counters are not monotonic")
    return 100 * (total - idle) / total


def parse_available_memory(text):
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            raw = line.split()
            if len(raw) == 3 and raw[1].isdigit() and raw[2] == "kB":
                return int(raw[1]) * 1024
    raise ValueError("/proc/meminfo has no valid MemAvailable value")


def parse_diskstats(text, whole_device_names):
    read_sectors = 0
    write_sectors = 0
    observed = set()
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 10 or fields[2] not in whole_device_names:
            continue
        try:
            read_sectors += int(fields[5])
            write_sectors += int(fields[9])
        except ValueError as error:
            raise ValueError("/proc/diskstats contains an invalid sector counter") from error
        observed.add(fields[2])
    if observed != set(whole_device_names):
        raise ValueError("/proc/diskstats is missing a selected whole device")
    return {"disk_read_bytes": read_sectors * 512, "disk_write_bytes": write_sectors * 512}


def _run(command):
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def sample_once(
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
    gpu = parse_gpu_csv(
        runner(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(GPU_FIELDS)}",
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
    disk = parse_diskstats(reader("/proc/diskstats"), device_names)
    sample = {
        "monotonic_ns": monotonic_ns(),
        "utc": utc_now().isoformat(),
        **gpu,
        "cpu_utilization_percent": cpu_utilization(previous_cpu, cpu),
        "load_average_1m": _number(load_raw[0], "load average"),
        "available_memory_bytes": parse_available_memory(reader("/proc/meminfo")),
        **disk,
        "gpu_process_pids": processes,
        "benchmark_pids": sorted(set(benchmark_pids)),
    }
    return sample, cpu


def sample_series(
    count,
    benchmark_pids,
    *,
    interval_seconds=1,
    sampler=sample_once,
    sleeper=time.sleep,
    monotonic=time.monotonic,
    **sample_kwargs,
):
    if isinstance(count, bool) or not isinstance(count, int) or count < 2 or interval_seconds != 1:
        raise ValueError("systems telemetry sampling requires at least two samples at 1 Hz")
    samples = []
    previous_cpu = None
    deadline = monotonic()
    for index in range(count):
        if index:
            deadline += interval_seconds
            delay = deadline - monotonic()
            if delay > 0:
                sleeper(delay)
        sample, previous_cpu = sampler(
            previous_cpu,
            benchmark_pids,
            **sample_kwargs,
        )
        samples.append(sample)
    return validate_samples(samples, GPU_UUID)


def serialize_sample_series(protocol_path, count, benchmark_pids, device_names, **sample_kwargs):
    protocol_path = Path(protocol_path).expanduser().resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != "speck_paper_finalist_systems_protocol"
        or protocol.get("format_version") != 2
        or protocol.get("telemetry", {}).get("sampling_hz") != 1
        or protocol.get("hardware_and_software_lock", {}).get("gpu_uuid") != GPU_UUID
    ):
        raise ValueError("telemetry sampler requires the frozen systems protocol")
    devices = set(device_names)
    if not devices or any(not isinstance(name, str) or not name for name in devices):
        raise ValueError("telemetry sampler requires explicit valid disk devices")
    return {
        "format": "speck_paper_finalist_systems_telemetry_samples",
        "format_version": 1,
        "protocol_sha256": file_sha256(protocol_path),
        "sampling_hz": 1,
        "disk_devices": sorted(devices),
        "samples": sample_series(
            count,
            benchmark_pids,
            device_names=devices,
            **sample_kwargs,
        ),
    }
