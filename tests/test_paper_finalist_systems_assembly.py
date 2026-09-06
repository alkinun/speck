import json
from copy import deepcopy
from pathlib import Path

import pytest

from speck.paper_finalist_systems_analysis import analyze_systems
from speck.paper_finalist_systems_assembly import assemble_block, assemble_trial
from speck.paper_finalist_systems_engine import file_sha256 as engine_sha256
from speck.paper_finalist_systems_telemetry import atomic_json, file_sha256
from speck.paper_finalist_systems_workload import build_trial_plan
from tests.test_paper_finalist_systems_telemetry import sample

root = Path(__file__).parents[1]
protocol_path = root / "research" / "paper-1" / "finalist_systems_v2.json"
memory_path = root / "research" / "paper-1" / "finalist_systems_memory_v1.json"
qualification_path = root / "results" / "Speck-Paper1" / "finalist-qualification-v1.json"
engine_module = root / "speck" / "paper_finalist_systems_engine.py"
protocol = json.loads(protocol_path.read_text(encoding="utf-8"))


def plans(tmp_path):
    return build_trial_plan(protocol_path, qualification_path, tmp_path / "planned")["trials"][:2]


def markers():
    return {
        "process_start": 120_000_000_000,
        "checkpoint_loaded": 122_000_000_000,
        "batches_materialized": 124_000_000_000,
        "warmup_start": 125_000_000_000,
        "warmup_end": 130_000_000_000,
        "measured_start": 130_000_000_000,
        "measured_end": 150_000_000_000,
        "process_end": 160_000_000_000,
    }


def trace_samples(pid):
    values = []
    for second in range(281):
        power = 30 if second <= 130 or second >= 150 else 200
        value = sample(second, power)
        value["memory_used_bytes"] = (1000 + (500 if 130 <= second <= 150 else 0)) * 1_048_576
        value["benchmark_pids"] = [pid] if 120 <= second <= 160 else []
        value["gpu_process_pids"] = [pid] if 130 <= second <= 150 else []
        values.append(value)
    return values


def write_trial_inputs(tmp_path, trial, *, fingerprint="a" * 64, pid=123):
    directory = tmp_path / trial["role"]
    directory.mkdir()
    engine = {
        "status": "complete_unintegrated",
        "arm_id": trial["arm_id"],
        "run": trial["run"],
        "checkpoint_step": 23496,
        "warmup_optimizer_steps": 10,
        "measured_optimizer_steps": 30,
        "measured_tokens": 1_966_080,
        "measured_wall_seconds": 10 if trial["role"] == "control" else 8,
        "phase_markers": markers(),
        "peak_allocated_bytes": 10_000,
        "peak_reserved_bytes": 12_000,
        "terminal_learning_rate": 0.00015,
        "paired_batches": {
            "sha256": fingerprint,
            "microbatches": 161,
            "optimizer_steps": 40,
            "accumulation_steps": 4,
            "materialized_device": "cpu",
            "transfer_device": "cuda",
            "H2D_inside_optimizer_windows": True,
        },
        "non_finite_steps": 0,
        "OOM": False,
        "validation_executed": False,
        "checkpoint_write_executed": False,
        "summary_write_executed": False,
        "tracking_initialized": False,
        "model_or_optimizer_output_persisted": False,
    }
    engine_path = directory / "engine.json"
    atomic_json(engine_path, engine)
    intervals = {
        "pre_idle_start_ns": 0,
        "pre_idle_end_ns": 120_000_000_000,
        "measured_start_ns": 130_000_000_000,
        "measured_end_ns": 150_000_000_000,
        "post_idle_start_ns": 160_000_000_000,
        "post_idle_end_ns": 280_000_000_000,
    }
    trace = {
        "format": "speck_paper_finalist_systems_telemetry_trace",
        "format_version": 1,
        "protocol_sha256": file_sha256(protocol_path),
        "memory_supplement_sha256": file_sha256(memory_path),
        "samples": trace_samples(pid),
        "intervals": intervals,
        "phase_markers": markers(),
    }
    trace_path = directory / "trace.json"
    atomic_json(trace_path, trace)
    identities = {
        "git_commit": "a" * 40,
        "python": "3.14",
        "pytorch": "2.9",
        "cuda_runtime": "13.0",
        "cuda_driver": "610.43.03",
        "fla": "0.4",
        "triton": "3.5",
        "model_config_sha256": file_sha256(Path(trial["experiment"]) / "model.json"),
        "benchmark_engine_sha256": engine_sha256(engine_module),
    }
    attestation = {
        "format": "speck_paper_finalist_systems_runtime_attestation",
        "format_version": 1,
        "status": "qualified",
        "protocol_sha256": file_sha256(protocol_path),
        "engine_result_sha256": file_sha256(engine_path),
        "trace_sha256": file_sha256(trace_path),
        "run": trial["run"],
        "benchmark_pid": pid,
        "compiled_CUDA": True,
        "kernel_fallback": False,
        "OOM": False,
        "cuda_phase_seconds": {"forward": 3.0, "backward": 4.0, "optimizer": 1.0},
        "software_identities": identities,
    }
    attestation_path = directory / "runtime.json"
    atomic_json(attestation_path, attestation)
    return engine_path, trace_path, attestation_path


def assemble_pair(tmp_path, **kwargs):
    outputs = []
    for trial in plans(tmp_path):
        engine, trace, runtime = write_trial_inputs(tmp_path, trial, **kwargs)
        result = assemble_trial(protocol_path, memory_path, trial, engine, trace, runtime)
        path = tmp_path / f"{trial['role']}-trial.json"
        atomic_json(path, result)
        outputs.append(path)
    return outputs


def rehash_trace_attestation(runtime, trace):
    value = json.loads(runtime.read_text(encoding="utf-8"))
    value["trace_sha256"] = file_sha256(trace)
    atomic_json(runtime, value)


def test_trial_and_block_assembly_feed_the_frozen_analyzer_contract(tmp_path):
    trial_paths = assemble_pair(tmp_path)
    block = assemble_block(protocol_path, 0, trial_paths)
    assert block["status"] == "complete_qualified"
    assert block["paired_batch_sha256"] == "a" * 64
    assert block["trials"]["control"]["peak_nvml_used_bytes"] == 1500 * 1_048_576
    assert block["trials"]["control"]["kernel_fallback"] is False
    assert block["trials"]["candidate"]["measured_wall_seconds"] == 8
    block_path = tmp_path / "block.json"
    atomic_json(block_path, block)
    analysis = analyze_systems(protocol_path, [block_path])
    assert analysis["status"] == "incomplete_failed_no_training_systems_claim"
    assert analysis["valid_blocks"] == [0]


def test_block_assembly_retains_paired_fingerprint_mismatch(tmp_path):
    trial_paths = assemble_pair(tmp_path)
    candidate = json.loads(trial_paths[1].read_text(encoding="utf-8"))
    candidate["paired_batches"]["sha256"] = "b" * 64
    atomic_json(trial_paths[1], candidate)
    block = assemble_block(protocol_path, 0, trial_paths)
    assert block["status"] == "failed_retained"
    assert block["failure"] == "paired_batch_fingerprint_mismatch"
    assert block["replacement_or_retry_authorized"] is False


def test_block_assembly_retains_missing_trial(tmp_path):
    trial_paths = assemble_pair(tmp_path)
    block = assemble_block(protocol_path, 0, trial_paths[:1])
    assert block["status"] == "failed_retained"
    assert block["failure"] == "missing_or_duplicate_trial"


def test_trial_assembly_rejects_phase_mismatch(tmp_path):
    trial = plans(tmp_path)[0]
    engine, trace, runtime = write_trial_inputs(tmp_path, trial)
    value = json.loads(trace.read_text(encoding="utf-8"))
    value["phase_markers"]["measured_end"] += 1
    atomic_json(trace, value)
    with pytest.raises(ValueError, match="phase markers differ"):
        assemble_trial(protocol_path, memory_path, trial, engine, trace, runtime)


def test_trial_assembly_rejects_runtime_fallback(tmp_path):
    trial = plans(tmp_path)[0]
    engine, trace, runtime = write_trial_inputs(tmp_path, trial)
    value = json.loads(runtime.read_text(encoding="utf-8"))
    value["kernel_fallback"] = True
    atomic_json(runtime, value)
    with pytest.raises(ValueError, match="runtime attestation"):
        assemble_trial(protocol_path, memory_path, trial, engine, trace, runtime)


def test_trial_assembly_rejects_missing_memory_samples(tmp_path):
    trial = plans(tmp_path)[0]
    engine, trace, runtime = write_trial_inputs(tmp_path, trial)
    value = json.loads(trace.read_text(encoding="utf-8"))
    for item in value["samples"]:
        item.pop("memory_used_bytes")
    atomic_json(trace, value)
    rehash_trace_attestation(runtime, trace)
    with pytest.raises(ValueError, match="memory_used_bytes"):
        assemble_trial(protocol_path, memory_path, trial, engine, trace, runtime)


def test_trial_assembly_rejects_unbound_process(tmp_path):
    trial = plans(tmp_path)[0]
    engine, trace, runtime = write_trial_inputs(tmp_path, trial)
    value = json.loads(trace.read_text(encoding="utf-8"))
    value["samples"][140]["gpu_process_pids"] = []
    atomic_json(trace, value)
    rehash_trace_attestation(runtime, trace)
    with pytest.raises(ValueError, match="benchmark process"):
        assemble_trial(protocol_path, memory_path, trial, engine, trace, runtime)


def test_trial_assembly_rejects_incomplete_batch_identity(tmp_path):
    trial = plans(tmp_path)[0]
    engine, trace, runtime = write_trial_inputs(tmp_path, trial)
    value = json.loads(engine.read_text(encoding="utf-8"))
    value["paired_batches"].pop("sha256")
    atomic_json(engine, value)
    runtime_value = json.loads(runtime.read_text(encoding="utf-8"))
    runtime_value["engine_result_sha256"] = file_sha256(engine)
    atomic_json(runtime, runtime_value)
    with pytest.raises(ValueError, match="paired-batch evidence"):
        assemble_trial(protocol_path, memory_path, trial, engine, trace, runtime)


def test_trial_assembly_rejects_hot_start(tmp_path):
    trial = plans(tmp_path)[0]
    engine, trace, runtime = write_trial_inputs(tmp_path, trial)
    value = json.loads(trace.read_text(encoding="utf-8"))
    value["samples"][120]["temperature_c"] = 46
    atomic_json(trace, value)
    rehash_trace_attestation(runtime, trace)
    with pytest.raises(ValueError, match="temperature exceeds"):
        assemble_trial(protocol_path, memory_path, trial, engine, trace, runtime)


def test_block_assembly_retains_common_software_mismatch(tmp_path):
    trial_paths = assemble_pair(tmp_path)
    candidate = deepcopy(json.loads(trial_paths[1].read_text(encoding="utf-8")))
    candidate["software_identities"]["pytorch"] = "different"
    atomic_json(trial_paths[1], candidate)
    block = assemble_block(protocol_path, 0, trial_paths)
    assert block["status"] == "failed_retained"
    assert block["failure"] == "runtime_software_identity_mismatch"


def test_block_assembly_retains_paired_temperature_mismatch(tmp_path):
    trial_paths = assemble_pair(tmp_path)
    candidate = json.loads(trial_paths[1].read_text(encoding="utf-8"))
    candidate["start_temperature_c"] = 41
    atomic_json(trial_paths[1], candidate)
    block = assemble_block(protocol_path, 0, trial_paths)
    assert block["status"] == "failed_retained"
    assert block["failure"] == "paired_start_temperature_mismatch"
