"""Assemble finalist systems engine, telemetry, and runtime evidence into paired blocks."""

import hashlib
import json
import math
from pathlib import Path

from speck.paper_finalist_systems_sampler_memory import (
    load_memory_contract,
    peak_memory_used,
)
from speck.paper_finalist_systems_telemetry import integrate_trace

SOFTWARE_IDENTITY_FIELDS = {
    "git_commit",
    "python",
    "pytorch",
    "cuda_runtime",
    "cuda_driver",
    "fla",
    "triton",
    "model_config_sha256",
    "benchmark_engine_sha256",
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


def reference(path):
    return {"path": str(path), "sha256": file_sha256(path)}


def _start_temperature(samples, start_ns):
    before = [sample for sample in samples if sample["monotonic_ns"] <= start_ns]
    after = [sample for sample in samples if sample["monotonic_ns"] >= start_ns]
    if not before or not after:
        raise ValueError("systems trial start temperature is not bracketed")
    return max(before[-1]["temperature_c"], after[0]["temperature_c"])


def _validate_process_binding(samples, markers, benchmark_pid):
    process_samples = [
        sample
        for sample in samples
        if markers["process_start"] <= sample["monotonic_ns"] <= markers["process_end"]
    ]
    measured_samples = [
        sample
        for sample in samples
        if markers["measured_start"] <= sample["monotonic_ns"] <= markers["measured_end"]
    ]
    if (
        len(process_samples) < 2
        or len(measured_samples) < 2
        or any(benchmark_pid not in sample["benchmark_pids"] for sample in process_samples)
        or any(benchmark_pid not in sample["gpu_process_pids"] for sample in measured_samples)
    ):
        raise ValueError("systems trial telemetry is not bound to the benchmark process")


def _validate_runtime_attestation(attestation, engine, trace, trial, protocol_sha):
    identities = attestation.get("software_identities", {})
    phase = attestation.get("cuda_phase_seconds", {})
    if (
        attestation.get("format") != "speck_paper_finalist_systems_runtime_attestation"
        or attestation.get("format_version") != 1
        or attestation.get("status") != "qualified"
        or attestation.get("protocol_sha256") != protocol_sha
        or attestation.get("engine_result_sha256") != file_sha256(engine[0])
        or attestation.get("trace_sha256") != file_sha256(trace[0])
        or attestation.get("run") != trial["run"]
        or attestation.get("compiled_CUDA") is not True
        or attestation.get("kernel_fallback") is not False
        or attestation.get("OOM") is not False
        or set(phase) != {"forward", "backward", "optimizer"}
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0
            for value in phase.values()
        )
        or set(identities) != SOFTWARE_IDENTITY_FIELDS
        or any(not isinstance(value, str) or not value for value in identities.values())
        or identities.get("benchmark_engine_sha256")
        != file_sha256(Path(__file__).with_name("paper_finalist_systems_engine.py"))
    ):
        raise ValueError("systems trial runtime attestation is invalid")
    return identities, phase


def assemble_trial(
    protocol_path,
    memory_contract_path,
    trial,
    engine_result_path,
    trace_path,
    runtime_attestation_path,
):
    protocol_path, protocol = load_object(protocol_path)
    memory_contract_path, _ = load_memory_contract(memory_contract_path)
    engine = load_object(engine_result_path)
    trace = load_object(trace_path)
    attestation_path, attestation = load_object(runtime_attestation_path)
    protocol_sha = file_sha256(protocol_path)
    memory_sha = file_sha256(memory_contract_path)
    engine_value = engine[1]
    trace_value = trace[1]
    if (
        engine_value.get("status") != "complete_unintegrated"
        or engine_value.get("arm_id") != trial["arm_id"]
        or engine_value.get("run") != trial["run"]
        or engine_value.get("checkpoint_step") != trial["checkpoint_step"]
        or engine_value.get("warmup_optimizer_steps") != trial["warmup_optimizer_steps"]
        or engine_value.get("measured_optimizer_steps") != trial["measured_optimizer_steps"]
        or engine_value.get("measured_tokens") != trial["measured_tokens"]
        or engine_value.get("non_finite_steps") != 0
        or engine_value.get("OOM") is not False
        or engine_value.get("validation_executed") is not False
        or engine_value.get("checkpoint_write_executed") is not False
        or engine_value.get("summary_write_executed") is not False
        or engine_value.get("tracking_initialized") is not False
        or engine_value.get("model_or_optimizer_output_persisted") is not False
    ):
        raise ValueError("systems engine result does not match its trial plan")
    markers = engine_value.get("phase_markers")
    if trace_value.get("phase_markers") != markers:
        raise ValueError("systems engine and telemetry phase markers differ")
    if (
        trace_value.get("protocol_sha256") != protocol_sha
        or trace_value.get("memory_supplement_sha256") != memory_sha
    ):
        raise ValueError("systems telemetry trace does not match its protocol supplements")
    identities, phase_seconds = _validate_runtime_attestation(
        attestation,
        engine,
        trace,
        trial,
        protocol_sha,
    )
    measured_wall = engine_value.get("measured_wall_seconds")
    paired_batches = engine_value.get("paired_batches", {})
    fingerprint = paired_batches.get("sha256", "")
    if (
        isinstance(measured_wall, bool)
        or not isinstance(measured_wall, (int, float))
        or not math.isfinite(measured_wall)
        or measured_wall <= 0
        or sum(phase_seconds.values()) > measured_wall
        or engine_value.get("terminal_learning_rate") != 0.00015
        or not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
        or paired_batches.get("microbatches") != 161
        or paired_batches.get("optimizer_steps") != 40
        or paired_batches.get("accumulation_steps") != 4
        or paired_batches.get("materialized_device") != "cpu"
        or paired_batches.get("H2D_inside_optimizer_windows") is not True
    ):
        raise ValueError("systems engine timing or paired-batch evidence is invalid")
    model_config = Path(trial["experiment"]) / "model.json"
    if not model_config.is_file() or identities["model_config_sha256"] != file_sha256(model_config):
        raise ValueError("systems runtime attestation has the wrong model config")
    samples = trace_value.get("samples")
    benchmark_pid = attestation.get("benchmark_pid")
    if isinstance(benchmark_pid, bool) or not isinstance(benchmark_pid, int) or benchmark_pid < 1:
        raise ValueError("systems runtime attestation has an invalid benchmark PID")
    _validate_process_binding(samples, markers, benchmark_pid)
    integration = integrate_trace(protocol_path, trace[0])
    if integration.get("qualified") is not True:
        raise ValueError("systems telemetry energy integration is not qualified")
    peak_nvml = peak_memory_used(samples, markers["measured_start"], markers["measured_end"])
    start_temperature = _start_temperature(samples, markers["process_start"])
    if start_temperature > protocol["thermal_and_idle_design"]["maximum_trial_start_temperature_c"]:
        raise ValueError("systems trial start temperature exceeds the frozen limit")
    peak_allocated = engine_value.get("peak_allocated_bytes")
    peak_reserved = engine_value.get("peak_reserved_bytes")
    if (
        isinstance(peak_allocated, bool)
        or not isinstance(peak_allocated, int)
        or peak_allocated < 1
        or isinstance(peak_reserved, bool)
        or not isinstance(peak_reserved, int)
        or peak_reserved < peak_allocated
    ):
        raise ValueError("systems engine peak torch memory is invalid")
    return {
        "format": "speck_paper_finalist_systems_trial_result",
        "format_version": 1,
        "status": "complete_qualified",
        "block": trial["block"],
        "pair": trial["pair"],
        "position": trial["position"],
        "role": trial["role"],
        "arm_id": trial["arm_id"],
        "run": trial["run"],
        "protocol_sha256": protocol_sha,
        "memory_supplement_sha256": memory_sha,
        "engine_result": reference(engine[0]),
        "telemetry_trace": reference(trace[0]),
        "runtime_attestation": reference(attestation_path),
        "measured_optimizer_steps": trial["measured_optimizer_steps"],
        "measured_tokens": trial["measured_tokens"],
        "measured_wall_seconds": measured_wall,
        "gross_board_energy_joules": integration["gross_board_energy_joules"],
        "incremental_board_energy_joules": integration["incremental_board_energy_joules"],
        "start_temperature_c": start_temperature,
        "telemetry": integration["telemetry"],
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_nvml_used_bytes": peak_nvml,
        "non_finite_steps": 0,
        "OOM": False,
        "kernel_fallback": False,
        "paired_batches": paired_batches,
        "cuda_phase_seconds": phase_seconds,
        "software_identities": identities,
    }


def _failed_block(protocol_path, expected, references, reason):
    return {
        "format": "speck_paper_finalist_systems_block_result",
        "format_version": 1,
        "status": "failed_retained",
        "protocol_sha256": file_sha256(protocol_path),
        "block": expected["block"],
        "pair": expected["pair"],
        "trial_order": expected["trial_order"],
        "failure": reason,
        "trial_references": references,
        "replacement_or_retry_authorized": False,
    }


def assemble_block(protocol_path, block, trial_result_paths):
    protocol_path, protocol = load_object(protocol_path)
    expected = next(
        (value for value in protocol["paired_blocks"] if value["block"] == block),
        None,
    )
    if expected is None:
        raise ValueError("systems block is outside the frozen protocol")
    reports = []
    references = []
    for path in trial_result_paths:
        path, report = load_object(path)
        reports.append(report)
        references.append({**reference(path), "role": report.get("role")})
    by_role = {report.get("role"): report for report in reports}
    if len(reports) != 2 or set(by_role) != {"control", "candidate"}:
        return _failed_block(protocol_path, expected, references, "missing_or_duplicate_trial")
    if any(report.get("status") != "complete_qualified" for report in reports):
        return _failed_block(protocol_path, expected, references, "retained_trial_failure")
    for position, role in enumerate(expected["trial_order"]):
        report = by_role[role]
        if (
            report.get("protocol_sha256") != file_sha256(protocol_path)
            or report.get("block") != block
            or report.get("pair") != expected["pair"]
            or report.get("position") != position
            or report.get("arm_id") != protocol["arms"][role]
        ):
            raise ValueError("systems trial result does not match its frozen block position")
    if by_role["control"].get("paired_batches", {}).get("sha256") != by_role["candidate"].get(
        "paired_batches", {}
    ).get("sha256"):
        return _failed_block(
            protocol_path, expected, references, "paired_batch_fingerprint_mismatch"
        )
    common_fields = SOFTWARE_IDENTITY_FIELDS - {"model_config_sha256"}
    if any(
        by_role["control"]["software_identities"][field]
        != by_role["candidate"]["software_identities"][field]
        for field in common_fields
    ):
        return _failed_block(
            protocol_path, expected, references, "runtime_software_identity_mismatch"
        )
    if (
        abs(by_role["control"]["start_temperature_c"] - by_role["candidate"]["start_temperature_c"])
        > protocol["thermal_and_idle_design"]["maximum_within_pair_start_temperature_difference_c"]
    ):
        return _failed_block(
            protocol_path,
            expected,
            references,
            "paired_start_temperature_mismatch",
        )
    return {
        "format": "speck_paper_finalist_systems_block_result",
        "format_version": 1,
        "status": "complete_qualified",
        "protocol_sha256": file_sha256(protocol_path),
        "block": expected["block"],
        "pair": expected["pair"],
        "trial_order": expected["trial_order"],
        "trial_references": references,
        "paired_batch_sha256": by_role["control"]["paired_batches"]["sha256"],
        "trials": by_role,
        "replacement_or_retry_authorized": False,
    }
