from pathlib import Path
from types import SimpleNamespace

import pytest

from speck.paper_finalist_systems_assembly import assemble_trial
from speck.paper_finalist_systems_runtime import (
    build_runtime_attestation,
    collect_software_identities,
    parse_driver_row,
)
from speck.paper_finalist_systems_telemetry import GPU_UUID, atomic_json, file_sha256
from tests.test_paper_finalist_systems_assembly import plans, write_trial_inputs

root = Path(__file__).parents[1]
protocol_path = root / "research" / "paper-1" / "finalist_systems_v2.json"
memory_path = root / "research" / "paper-1" / "finalist_systems_memory_v1.json"


class TorchFixture:
    __version__ = "2.9.1+cu128"
    version = SimpleNamespace(cuda="12.8")


def identity_runner(command):
    if command[-2:] == ["status", "--porcelain"]:
        return ""
    if command[-2:] == ["rev-parse", "HEAD"]:
        return "a" * 40 + "\n"
    if command[0] == "nvidia-smi":
        return f"{GPU_UUID}, 610.43.03\n"
    raise AssertionError(command)


def version_lookup(name):
    return {"flash-linear-attention": "0.5.0", "triton": "3.5.0"}[name]


def identity_kwargs():
    return {
        "runner": identity_runner,
        "version_lookup": version_lookup,
        "torch_module": TorchFixture,
        "python_version": lambda: "3.10.20",
    }


def write_probe(path, trial, engine, trace, **changes):
    probe = {
        "format": "speck_paper_finalist_systems_runtime_probe",
        "format_version": 1,
        "status": "qualified",
        "protocol_sha256": file_sha256(protocol_path),
        "engine_result_sha256": file_sha256(engine),
        "trace_sha256": file_sha256(trace),
        "run": trial["run"],
        "benchmark_pid": 123,
        "compiled_CUDA": True,
        "compiled_graphs": 1,
        "graph_breaks": 0,
        "eager_fallbacks": 0,
        "OOM": False,
        "cuda_phase_seconds": {"forward": 3.0, "backward": 4.0, "optimizer": 1.0},
    }
    probe.update(changes)
    atomic_json(path, probe)


def test_runtime_identity_collects_exact_GPU_environment_vector(tmp_path):
    model = tmp_path / "model.json"
    engine = tmp_path / "engine.py"
    model.write_text("{}\n", encoding="utf-8")
    engine.write_text("pass\n", encoding="utf-8")
    identities = collect_software_identities(
        root,
        model,
        engine,
        **identity_kwargs(),
    )
    assert identities == {
        "git_commit": "a" * 40,
        "python": "3.10.20",
        "pytorch": "2.9.1+cu128",
        "cuda_runtime": "12.8",
        "cuda_driver": "610.43.03",
        "fla": "0.5.0",
        "triton": "3.5.0",
        "model_config_sha256": file_sha256(model),
        "benchmark_engine_sha256": file_sha256(engine),
    }


def test_runtime_identity_rejects_dirty_repository(tmp_path):
    model = tmp_path / "model.json"
    engine = tmp_path / "engine.py"
    model.write_text("{}", encoding="utf-8")
    engine.write_text("pass", encoding="utf-8")

    def dirty_runner(command):
        if command[-2:] == ["status", "--porcelain"]:
            return " M file\n"
        return identity_runner(command)

    with pytest.raises(ValueError, match="clean repository"):
        collect_software_identities(
            root,
            model,
            engine,
            **{**identity_kwargs(), "runner": dirty_runner},
        )


def test_runtime_identity_rejects_wrong_GPU_driver_row():
    with pytest.raises(ValueError, match="UUID or driver"):
        parse_driver_row("GPU-wrong, 610.43.03\n")


def test_runtime_attestation_feeds_trial_assembler(tmp_path):
    trial = plans(tmp_path)[0]
    engine, trace, old_runtime = write_trial_inputs(tmp_path, trial)
    old_runtime.unlink()
    probe = tmp_path / "runtime-probe.json"
    write_probe(probe, trial, engine, trace)
    attestation = build_runtime_attestation(
        protocol_path,
        trial,
        engine,
        trace,
        probe,
        identity_kwargs=identity_kwargs(),
    )
    runtime = tmp_path / "runtime.json"
    atomic_json(runtime, attestation)
    assembled = assemble_trial(protocol_path, memory_path, trial, engine, trace, runtime)
    assert assembled["status"] == "complete_qualified"
    assert assembled["kernel_fallback"] is False
    assert assembled["software_identities"]["fla"] == "0.5.0"


@pytest.mark.parametrize(
    "change",
    (
        {"graph_breaks": 1},
        {"eager_fallbacks": 1},
        {"compiled_graphs": 0},
        {"OOM": True},
    ),
)
def test_runtime_attestation_rejects_unqualified_probe(tmp_path, change):
    trial = plans(tmp_path)[0]
    engine, trace, _ = write_trial_inputs(tmp_path, trial)
    probe = tmp_path / "runtime-probe.json"
    write_probe(probe, trial, engine, trace, **change)
    with pytest.raises(ValueError, match="runtime probe"):
        build_runtime_attestation(
            protocol_path,
            trial,
            engine,
            trace,
            probe,
            identity_kwargs=identity_kwargs(),
        )


def test_runtime_attestation_rejects_phase_time_above_wall(tmp_path):
    trial = plans(tmp_path)[0]
    engine, trace, _ = write_trial_inputs(tmp_path, trial)
    probe = tmp_path / "runtime-probe.json"
    write_probe(
        probe,
        trial,
        engine,
        trace,
        cuda_phase_seconds={"forward": 8.0, "backward": 8.0, "optimizer": 8.0},
    )
    with pytest.raises(ValueError, match="runtime probe"):
        build_runtime_attestation(
            protocol_path,
            trial,
            engine,
            trace,
            probe,
            identity_kwargs=identity_kwargs(),
        )
