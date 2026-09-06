"""Orchestrate one frozen finalist systems block through an injectable live adapter."""

import json
from pathlib import Path

from speck.paper_finalist_systems_telemetry import GPU_UUID


def load_object(path):
    path = Path(path).expanduser().resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return path, value


def _idle_interval(value, seconds, name):
    if (
        not isinstance(value, dict)
        or isinstance(value.get("start_ns"), bool)
        or not isinstance(value.get("start_ns"), int)
        or isinstance(value.get("end_ns"), bool)
        or not isinstance(value.get("end_ns"), int)
        or value["end_ns"] - value["start_ns"] != seconds * 1_000_000_000
    ):
        raise ValueError(f"systems orchestration {name} idle interval changed")
    return value


def _gate(value, maximum_temperature):
    if (
        not isinstance(value, dict)
        or value.get("gpu_uuid") != GPU_UUID
        or isinstance(value.get("temperature_c"), bool)
        or not isinstance(value.get("temperature_c"), (int, float))
        or value["temperature_c"] > maximum_temperature
        or value.get("gpu_utilization_percent") != 0
        or value.get("compute_process_pids") != []
    ):
        raise ValueError("systems orchestration trial live gate failed")
    return value


def _failure(adapter, block, events, error, sampler, stopped):
    cleanup_error = None
    if sampler is not None and not stopped:
        try:
            adapter.stop_sampler(sampler)
            events.append("sampler_stop_after_failure")
        except Exception as stop_error:
            cleanup_error = f"{type(stop_error).__name__}: {stop_error}"
    failure = {
        "format": "speck_paper_finalist_systems_orchestration_failure",
        "format_version": 1,
        "status": "failed_retained",
        "block": block,
        "failure_type": type(error).__name__,
        "failure": str(error),
        "sampler_cleanup_failure": cleanup_error,
        "events": events,
        "replacement_or_retry_authorized": False,
        "successor_block_authorized": False,
    }
    reference = adapter.retain_failure(failure)
    return {**failure, "failure_reference": reference}


def orchestrate_block(protocol_path, workload_plan, block, adapter, program_started_ns):
    """Execute the frozen block state machine; all effects are delegated to the adapter."""

    _, protocol = load_object(protocol_path)
    if (
        protocol.get("format") != "speck_paper_finalist_systems_protocol"
        or protocol.get("format_version") != 2
        or workload_plan.get("format") != "speck_paper_finalist_systems_workload_plan"
        or workload_plan.get("execution_authorized") is not False
    ):
        raise ValueError("systems orchestration inputs are invalid")
    expected = next(
        (value for value in protocol["paired_blocks"] if value["block"] == block),
        None,
    )
    trials = sorted(
        (value for value in workload_plan["trials"] if value["block"] == block),
        key=lambda value: value["position"],
    )
    if (
        expected is None
        or len(trials) != 2
        or [trial["position"] for trial in trials] != [0, 1]
        or [trial["role"] for trial in trials] != expected["trial_order"]
        or any(trial.get("execution_authorized") is not False for trial in trials)
    ):
        raise ValueError("systems orchestration block plan changed")
    resource = protocol["resource_envelope"]
    thermal = protocol["thermal_and_idle_design"]
    started_ns = adapter.now_ns()
    if (
        isinstance(program_started_ns, bool)
        or not isinstance(program_started_ns, int)
        or started_ns < program_started_ns
        or (started_ns - program_started_ns) / 3.6e12 >= resource["maximum_total_elapsed_hours"]
    ):
        raise ValueError("systems orchestration elapsed budget is exhausted before the block")
    sampler = None
    stopped = False
    events = []
    try:
        sampler = adapter.start_sampler(block)
        events.append("sampler_start")
        pre_idle = _idle_interval(
            adapter.idle(sampler, thermal["pre_block_idle_seconds"], "pre_block"),
            thermal["pre_block_idle_seconds"],
            "pre_block",
        )
        events.append("pre_block_idle")
        gates = []
        engines = []
        for position, trial in enumerate(trials):
            if position:
                recovery = adapter.recover(
                    sampler,
                    thermal["maximum_trial_start_temperature_c"],
                    thermal["thermal_recovery_timeout_seconds"],
                )
                if (
                    not isinstance(recovery, dict)
                    or isinstance(recovery.get("elapsed_seconds"), bool)
                    or not isinstance(recovery.get("elapsed_seconds"), (int, float))
                    or not 0
                    <= recovery["elapsed_seconds"]
                    <= thermal["thermal_recovery_timeout_seconds"]
                    or recovery.get("end_temperature_c")
                    > thermal["maximum_trial_start_temperature_c"]
                ):
                    raise ValueError("systems orchestration thermal recovery failed")
                events.append("thermal_recovery")
            gate = _gate(
                adapter.live_gate(trial),
                thermal["maximum_trial_start_temperature_c"],
            )
            gates.append(gate)
            events.append(f"trial_{position}_gate")
            if (
                position
                and abs(gates[0]["temperature_c"] - gates[1]["temperature_c"])
                > thermal["maximum_within_pair_start_temperature_difference_c"]
            ):
                raise ValueError("systems orchestration paired start temperatures differ")
            engines.append(adapter.run_engine(trial, sampler))
            events.append(f"trial_{position}_engine")
        post_idle = _idle_interval(
            adapter.idle(sampler, thermal["post_block_idle_seconds"], "post_block"),
            thermal["post_block_idle_seconds"],
            "post_block",
        )
        events.append("post_block_idle")
        samples = adapter.stop_sampler(sampler)
        stopped = True
        events.append("sampler_stop")
        traces = adapter.build_traces(
            block,
            samples,
            pre_idle,
            post_idle,
            engines,
            gates,
        )
        events.append("trace_build")
        assembled = adapter.assemble_block(block, trials, engines, traces)
        events.append("block_assembly")
        if assembled.get("status") != "complete_qualified":
            raise ValueError("systems orchestration block assembly did not qualify")
        ended_ns = adapter.now_ns()
        elapsed_seconds = (ended_ns - program_started_ns) / 1e9
        if elapsed_seconds > resource["maximum_total_elapsed_hours"] * 3600:
            raise ValueError("systems orchestration exceeded the four-hour resource envelope")
        return {
            "format": "speck_paper_finalist_systems_orchestration_result",
            "format_version": 1,
            "status": "complete_qualified",
            "block": block,
            "events": events,
            "gates": gates,
            "pre_idle": pre_idle,
            "post_idle": post_idle,
            "engine_results": engines,
            "trace_results": traces,
            "block_result": assembled,
            "program_elapsed_seconds": elapsed_seconds,
            "replacement_or_retry_authorized": False,
            "successor_block_authorized": block < len(protocol["paired_blocks"]) - 1,
        }
    except Exception as error:
        return _failure(adapter, block, events, error, sampler, stopped)
