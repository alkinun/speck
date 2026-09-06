"""Build hash-bound runtime attestations for finalist systems trials."""

import csv
import importlib.metadata
import io
import json
import math
import platform
import subprocess
from pathlib import Path

import torch

from speck.paper_finalist_systems_assembly import SOFTWARE_IDENTITY_FIELDS
from speck.paper_finalist_systems_telemetry import GPU_UUID, file_sha256


def load_object(path):
    path = Path(path).expanduser().resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return path, value


def _run(command):
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def parse_driver_row(output):
    rows = list(csv.reader(io.StringIO(output), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 2:
        raise ValueError("runtime identity requires exactly one GPU driver row")
    uuid, driver = (value.strip() for value in rows[0])
    if uuid != GPU_UUID or not driver:
        raise ValueError("runtime identity GPU UUID or driver changed")
    return driver


def collect_software_identities(
    repository_root,
    model_config_path,
    engine_path,
    *,
    runner=_run,
    version_lookup=importlib.metadata.version,
    torch_module=torch,
    python_version=platform.python_version,
):
    repository_root = Path(repository_root).resolve()
    if runner(["git", "-C", str(repository_root), "status", "--porcelain"]).strip():
        raise ValueError("runtime identity requires a clean repository")
    commit = runner(["git", "-C", str(repository_root), "rev-parse", "HEAD"]).strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("runtime identity git commit is invalid")
    driver = parse_driver_row(
        runner(
            [
                "nvidia-smi",
                "--query-gpu=uuid,driver_version",
                "--format=csv,noheader,nounits",
            ]
        )
    )
    cuda = torch_module.version.cuda
    versions = {
        "python": python_version(),
        "pytorch": torch_module.__version__,
        "cuda_runtime": cuda,
        "cuda_driver": driver,
        "fla": version_lookup("flash-linear-attention"),
        "triton": version_lookup("triton"),
    }
    if any(not isinstance(value, str) or not value for value in versions.values()):
        raise ValueError("runtime identity requires GPU-environment package versions")
    model_config_path = Path(model_config_path).resolve()
    engine_path = Path(engine_path).resolve()
    if not model_config_path.is_file() or not engine_path.is_file():
        raise ValueError("runtime identity model config or engine is absent")
    identities = {
        "git_commit": commit,
        **versions,
        "model_config_sha256": file_sha256(model_config_path),
        "benchmark_engine_sha256": file_sha256(engine_path),
    }
    if set(identities) != SOFTWARE_IDENTITY_FIELDS:
        raise ValueError("runtime identity field set is incomplete")
    return identities


def validate_runtime_probe(probe, engine_path, trace_path, trial, protocol_sha):
    phase = probe.get("cuda_phase_seconds", {})
    engine = load_object(engine_path)[1]
    if (
        probe.get("format") != "speck_paper_finalist_systems_runtime_probe"
        or probe.get("format_version") != 1
        or probe.get("status") != "qualified"
        or probe.get("protocol_sha256") != protocol_sha
        or probe.get("engine_result_sha256") != file_sha256(engine_path)
        or probe.get("trace_sha256") != file_sha256(trace_path)
        or probe.get("run") != trial["run"]
        or probe.get("compiled_CUDA") is not True
        or isinstance(probe.get("compiled_graphs"), bool)
        or not isinstance(probe.get("compiled_graphs"), int)
        or probe.get("compiled_graphs") < 1
        or probe.get("graph_breaks") != 0
        or probe.get("eager_fallbacks") != 0
        or probe.get("OOM") is not False
        or set(phase) != {"forward", "backward", "optimizer"}
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in phase.values()
        )
        or sum(phase.values()) > engine.get("measured_wall_seconds", 0)
    ):
        raise ValueError("finalist systems runtime probe is invalid")
    pid = probe.get("benchmark_pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
        raise ValueError("finalist systems runtime probe PID is invalid")
    return phase, pid


def build_runtime_attestation(
    protocol_path,
    trial,
    engine_result_path,
    trace_path,
    runtime_probe_path,
    *,
    identity_collector=collect_software_identities,
    identity_kwargs=None,
):
    protocol_path, protocol = load_object(protocol_path)
    engine_result_path, engine = load_object(engine_result_path)
    trace_path, _ = load_object(trace_path)
    runtime_probe_path, probe = load_object(runtime_probe_path)
    protocol_sha = file_sha256(protocol_path)
    if (
        protocol.get("format") != "speck_paper_finalist_systems_protocol"
        or protocol.get("format_version") != 2
        or engine.get("run") != trial["run"]
        or engine.get("arm_id") != trial["arm_id"]
    ):
        raise ValueError("runtime attestation inputs do not match the frozen trial")
    phase, pid = validate_runtime_probe(
        probe,
        engine_result_path,
        trace_path,
        trial,
        protocol_sha,
    )
    identity_kwargs = identity_kwargs or {}
    identities = identity_collector(
        protocol_path.parents[2],
        Path(trial["experiment"]) / "model.json",
        Path(__file__).with_name("paper_finalist_systems_engine.py"),
        **identity_kwargs,
    )
    return {
        "format": "speck_paper_finalist_systems_runtime_attestation",
        "format_version": 1,
        "status": "qualified",
        "protocol_sha256": protocol_sha,
        "engine_result_sha256": file_sha256(engine_result_path),
        "trace_sha256": file_sha256(trace_path),
        "runtime_probe": {
            "path": str(runtime_probe_path),
            "sha256": file_sha256(runtime_probe_path),
        },
        "run": trial["run"],
        "benchmark_pid": pid,
        "compiled_CUDA": True,
        "kernel_fallback": False,
        "OOM": False,
        "cuda_phase_seconds": phase,
        "software_identities": identities,
    }
