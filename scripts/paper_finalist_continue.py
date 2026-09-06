"""Advance the frozen Paper 1 finalist sequence from successful file events."""

import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
import select
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "research/paper-1/experiment_program.json"
PLAN = ROOT / "research/paper-1/finalist_analysis_v2.json"
MATERIALIZATION_CONTRACT = ROOT / "research/paper-1/finalist_materialization_v1.json"
LAUNCH_CONTRACT = ROOT / "research/paper-1/finalist_launch_v1.json"
AUTOMATION = ROOT / "research/paper-1/finalist_automation_v1.json"
QUALIFICATION = ROOT / "results/Speck-Paper1/finalist-qualification-v1.json"
RUNTIME_PREFLIGHT = ROOT / "results/Speck-Paper1/finalist-preflight-v1.json"
RESULTS = ROOT / "results/Speck-Paper1/finalist-runs"
TRANSITIONS = ROOT / "results/Speck-Paper1/finalist-transitions"
TARGET = ROOT / "results/Speck-Paper1/finalist-time-to-quality-target.json"
ANALYSIS = ROOT / "results/Speck-Paper1/finalist-analysis.json"
CPU_ENV = ROOT / ".venv-paper1-cpu"
UV = Path("/home/alkin/.local/bin/uv")
LOCK = Path("/run/user") / str(os.getuid()) / "speck-paper1-finalist.lock"
VOLUME_UUID = "b64b59d1-ea2c-4206-9171-b7cd739f3eff"
GPU_UUID = "GPU-6e7f2d05-ac19-3812-ff41-33079fd67bfd"


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("run")
    finalize.add_argument("--training-unit", required=True)
    finalize.add_argument("--trigger-unit", required=True)
    launch = subparsers.add_parser("launch")
    launch.add_argument("run")
    return parser.parse_args(argv)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(command, *, env=None, check=True):
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=check,
        capture_output=True,
        text=True,
    )


def git_clean():
    if run(["git", "status", "--porcelain"]).stdout.strip():
        raise ValueError("automatic finalist continuation requires a clean repository")


def _qualification_runs():
    return {entry["run"]: entry for entry in load_object(QUALIFICATION)["runs"]}


def ordered_runs():
    declared = load_object(LAUNCH_CONTRACT)["execution_order"]
    runs = _qualification_runs()
    return [entry for entry in declared if entry in runs]


def control_runs():
    return [run_name for run_name in ordered_runs() if run_name.endswith("dense_global_param_match")]


def candidate_runs():
    return [run_name for run_name in ordered_runs() if run_name.endswith("five_cache_kda_gqa")]


def expected_next(evidence):
    controls = evidence.get("control_results", ())
    candidates = evidence.get("candidate_results", ())
    target = evidence.get("time_to_quality_target")
    analysis = evidence.get("analysis_result")
    if len(controls) < 6:
        if candidates or target is not None or analysis is not None:
            raise ValueError("finalist evidence violates control-first ordering")
        return control_runs()[len(controls)]
    if len(controls) != 6 or target is None:
        raise ValueError("finalist evidence is missing its six-control target lock")
    if len(candidates) < 6:
        if analysis is not None:
            raise ValueError("finalist analysis exists before every candidate")
        return candidate_runs()[len(candidates)]
    if len(candidates) != 6 or analysis is None:
        raise ValueError("finalist evidence is incomplete after all candidates")
    return None


def _unit_stem(run_name):
    pair = _qualification_runs()[run_name]["seed"], _qualification_runs()[run_name][
        "data_token_offset"
    ]
    pair_index = next(
        value["pair"]
        for value in load_object(MATERIALIZATION_CONTRACT)["pairs"]
        if (value["seed"], value["data_token_offset"]) == pair
    )
    arm = "control" if run_name.endswith("dense_global_param_match") else "candidate"
    return f"speck-paper1-finalist-{arm}-pair{pair_index}"


def validate_automation_contract():
    contract = load_object(AUTOMATION)
    if (
        contract.get("format") != "speck_paper_finalist_automation_contract"
        or contract.get("format_version") != 1
        or contract.get("status") != "frozen_before_any_finalist_output"
        or contract.get("implementation", {}).get("runner_sha256") != file_sha256(__file__)
        or contract.get("inputs", {}).get("launch_contract_sha256")
        != file_sha256(LAUNCH_CONTRACT)
        or contract.get("inputs", {}).get("analysis_plan_sha256") != file_sha256(PLAN)
        or contract.get("inputs", {}).get("materialization_contract_sha256")
        != file_sha256(MATERIALIZATION_CONTRACT)
        or contract.get("inputs", {}).get("qualification_sha256")
        != file_sha256(QUALIFICATION)
        or contract.get("inputs", {}).get("runtime_preflight_sha256")
        != file_sha256(RUNTIME_PREFLIGHT)
        or contract.get("execution_order") != ordered_runs()
    ):
        raise ValueError("finalist automation contract does not match its frozen inputs")
    return contract


def _cpu_command(*arguments):
    environment = os.environ.copy()
    environment["UV_PROJECT_ENVIRONMENT"] = str(CPU_ENV)
    environment["UV_NO_PROGRESS"] = "1"
    return run([str(UV), "run", "--extra", "cpu", "python", "-m", *arguments], env=environment)


def _stop_trigger(unit):
    result = run(["systemctl", "--user", "stop", unit], check=False)
    if result.returncode not in {0, 5}:
        raise RuntimeError(f"could not stop one-shot trigger {unit}: {result.stderr.strip()}")


def _wait_for_process_exit(unit, timeout_seconds=300):
    result = run(
        ["systemctl", "--user", "show", unit, "--property=MainPID", "--value"],
        check=False,
    )
    raw = result.stdout.strip()
    pid = int(raw) if raw.isdigit() else 0
    if not pid:
        return
    try:
        descriptor = os.pidfd_open(pid)
    except ProcessLookupError:
        return
    try:
        ready, _, _ = select.select([descriptor], [], [], timeout_seconds)
        if not ready:
            raise TimeoutError(f"training unit did not exit within {timeout_seconds} seconds")
    finally:
        os.close(descriptor)


def _result_path(run_name):
    return RESULTS / f"{run_name}.json"


def _result_reference(path, expected_pair):
    report = load_object(path)
    if (
        report.get("format") != "speck_paper_finalist_run_result"
        or report.get("format_version") != 2
        or report.get("status") != "complete_qualified"
        or report.get("pair", {}).get("pair") != expected_pair
        or report.get("training_tokens") != 1_539_833_856
        or report.get("non_finite_steps") != 0
    ):
        raise ValueError(f"collected finalist result is not qualified: {path}")
    return {
        "pair": expected_pair,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": file_sha256(path),
        "status": report["status"],
    }


def _collect(run_name, entry):
    output = _result_path(run_name)
    if output.exists():
        raise FileExistsError(f"finalist result already exists: {output}")
    _cpu_command(
        "scripts.paper_finalist_analyze",
        "collect",
        str(PLAN.relative_to(ROOT)),
        str(MATERIALIZATION_CONTRACT.relative_to(ROOT)),
        entry["experiment"],
        "--checkpoint-dir",
        entry["checkpoint_directory"],
        "--output",
        str(output.relative_to(ROOT)),
    )
    return output


def _lock_target(control_entries):
    if TARGET.exists():
        raise FileExistsError(f"finalist target lock already exists: {TARGET}")
    _cpu_command(
        "scripts.paper_finalist_analyze",
        "lock-target",
        str(PLAN.relative_to(ROOT)),
        str(MATERIALIZATION_CONTRACT.relative_to(ROOT)),
        *[entry["path"] for entry in control_entries],
        "--output",
        str(TARGET.relative_to(ROOT)),
    )
    target = load_object(TARGET)
    if target.get("status") != "locked_from_six_controls_before_candidates":
        raise ValueError("finalist control-only target lock is invalid")
    return {
        "path": TARGET.relative_to(ROOT).as_posix(),
        "sha256": file_sha256(TARGET),
        "status": target["status"],
    }


def _analyze(control_entries, candidate_entries, target_reference):
    if ANALYSIS.exists():
        raise FileExistsError(f"finalist analysis already exists: {ANALYSIS}")
    _cpu_command(
        "scripts.paper_finalist_analyze",
        "analyze",
        str(PLAN.relative_to(ROOT)),
        str(MATERIALIZATION_CONTRACT.relative_to(ROOT)),
        *[entry["path"] for entry in control_entries + candidate_entries],
        "--target-lock",
        target_reference["path"],
        "--output",
        str(ANALYSIS.relative_to(ROOT)),
    )
    report = load_object(ANALYSIS)
    if (
        report.get("status")
        != "complete_crossed_factor_finalist_language_evidence_no_standalone_promotion"
    ):
        raise ValueError("completed finalist analysis is invalid")
    return {
        "path": ANALYSIS.relative_to(ROOT).as_posix(),
        "sha256": file_sha256(ANALYSIS),
        "status": report["status"],
    }


def _commit(paths, message):
    run(["git", "add", *[str(Path(path).relative_to(ROOT)) for path in paths]])
    run(["git", "commit", "-m", message])


def _schedule(run_name, delay="15m"):
    unit = f"{_unit_stem(run_name)}-launch"
    command = [
        "systemd-run",
        "--user",
        f"--unit={unit}",
        "--collect",
        f"--on-active={delay}",
        "--timer-property=AccuracySec=1s",
        "--property=Restart=no",
        "--property=TimeoutStopSec=120",
        f"--working-directory={ROOT}",
        "/usr/bin/python3",
        str(Path(__file__).resolve()),
        "launch",
        run_name,
    ]
    completed = run(command)
    print(completed.stdout.strip(), flush=True)


def finalize(run_name, training_unit, trigger_unit):
    if run_name not in ordered_runs():
        raise ValueError("run is outside the finalist sequence")
    _stop_trigger(trigger_unit)
    _wait_for_process_exit(training_unit)
    with LOCK.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        validate_automation_contract()
        git_clean()
        program = load_object(PROGRAM)
        evidence = program["finalist_evidence"]
        if expected_next(evidence) != run_name:
            raise ValueError("completed finalist run is not the next frozen cell")
        entry = _qualification_runs()[run_name]
        result_path = _collect(run_name, entry)
        pair = next(
            value["pair"]
            for value in load_object(MATERIALIZATION_CONTRACT)["pairs"]
            if value["seed"] == entry["seed"]
            and value["data_token_offset"] == entry["data_token_offset"]
        )
        changed = [result_path, PROGRAM]
        next_run = None
        if run_name in control_runs():
            controls = evidence["control_results"]
            controls.append(_result_reference(result_path, pair))
            if len(controls) < 6:
                next_run = control_runs()[len(controls)]
                message = f"Register Paper 1 finalist control {len(controls) - 1}"
            else:
                evidence["time_to_quality_target"] = _lock_target(controls)
                changed.append(TARGET)
                next_run = candidate_runs()[0]
                message = "Lock Paper 1 finalist control target"
        else:
            candidates = evidence["candidate_results"]
            candidates.append(_result_reference(result_path, pair))
            if len(candidates) < 6:
                next_run = candidate_runs()[len(candidates)]
                message = f"Register Paper 1 finalist candidate {len(candidates) - 1}"
            else:
                evidence["analysis_result"] = _analyze(
                    evidence["control_results"],
                    candidates,
                    evidence["time_to_quality_target"],
                )
                changed.append(ANALYSIS)
                message = "Complete Paper 1 finalist paired analysis"
        evidence["next_run"] = next_run
        evidence["status"] = "complete" if next_run is None else "in_progress"
        atomic_json(PROGRAM, program)
        transition = {
            "format": "speck_paper_finalist_automatic_transition",
            "format_version": 2,
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "completed_run": run_name,
            "result": {
                "path": result_path.relative_to(ROOT).as_posix(),
                "sha256": file_sha256(result_path),
            },
            "next_run": next_run,
            "quality_dependent_branching": False,
            "trigger_disabled_before_collection": True,
            "polling": False,
            "automatic_retry": False,
            "automation_contract_sha256": file_sha256(AUTOMATION),
        }
        transition_path = TRANSITIONS / f"{run_name}.json"
        atomic_json(transition_path, transition)
        changed.append(transition_path)
        _cpu_command("scripts.paper_program_validate", "research/paper-1")
        _commit(changed, message)
        if next_run is not None:
            _schedule(next_run)


def _gpu_state():
    output = run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,memory.total,memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    ).stdout
    rows = list(csv.reader(io.StringIO(output), skipinitialspace=True))
    if len(rows) != 1:
        raise ValueError("finalist launch requires exactly one visible GPU")
    name, uuid, total_mib, used_mib, utilization, temperature = rows[0]
    processes = run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"]
    ).stdout.strip()
    state = {
        "name": name.strip(),
        "uuid": uuid.strip(),
        "total_mib": int(total_mib),
        "used_mib": int(used_mib),
        "utilization": int(utilization),
        "temperature": int(temperature),
        "compute_processes": [line for line in processes.splitlines() if line.strip()],
    }
    if (
        state["name"] != "NVIDIA GeForce RTX 3090"
        or state["uuid"] != GPU_UUID
        or state["total_mib"] < 24_576
        or state["utilization"] != 0
        or state["temperature"] > 50
        or state["compute_processes"]
    ):
        raise ValueError(f"finalist launch GPU gate failed: {state}")
    return state


def _install_watcher(run_name, summary_path, training_unit):
    unit = f"{_unit_stem(run_name)}-finalize"
    completed = run(
        [
            "systemd-run",
            "--user",
            f"--unit={unit}",
            "--collect",
            f"--path-property=PathExists={summary_path}",
            "--property=Restart=no",
            f"--working-directory={ROOT}",
            "/usr/bin/python3",
            str(Path(__file__).resolve()),
            "finalize",
            run_name,
            f"--training-unit={training_unit}",
            f"--trigger-unit={unit}.path",
        ]
    )
    print(completed.stdout.strip(), flush=True)


def _live_gate(run_name):
    if run(["systemctl", "--user", "is-active", "speck-helmet-download.service"], check=False).stdout.strip() == "active":
        raise ValueError("HELMET acquisition must be inactive during finalist training")
    state = _gpu_state()
    mount = run(["findmnt", "-no", "UUID,FSTYPE,OPTIONS", "/mnt/speck-data"]).stdout
    if (
        VOLUME_UUID not in mount
        or "ext4" not in mount
        or any(option not in mount for option in ("rw", "nosuid", "nodev", "noexec"))
    ):
        raise ValueError("finalist checkpoint volume identity or options changed")
    if shutil.disk_usage("/mnt/speck-data/speck").free < 25_769_803_776:
        raise ValueError("finalist checkpoint volume is below the frozen free-space floor")
    memory = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        memory[key] = int(raw.strip().split()[0]) * 1024
    if memory["MemAvailable"] < 12_884_901_888:
        raise ValueError("host available memory is below the finalist floor")
    runs = _qualification_runs()
    pending = ordered_runs()[ordered_runs().index(run_name) :]
    existing = [
        str(path)
        for pending_run in pending
        for path in (Path(runs[pending_run]["checkpoint_directory"]), _result_path(pending_run))
        if path.exists()
    ]
    if existing:
        raise FileExistsError(f"unfinished finalist output exists before launch: {existing}")
    return {
        "gpu": state,
        "mount": mount.strip(),
        "free_bytes": shutil.disk_usage("/mnt/speck-data/speck").free,
        "host_available_bytes": memory["MemAvailable"],
        "checkpoint_absent": True,
        "result_absent": True,
    }


def launch(run_name):
    if run_name not in ordered_runs():
        raise ValueError("only frozen finalist runs may be launched")
    with LOCK.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        validate_automation_contract()
        git_clean()
        program = load_object(PROGRAM)
        evidence = program["finalist_evidence"]
        if expected_next(evidence) != run_name or evidence.get("next_run") != run_name:
            raise ValueError("finalist program is not ready for this run")
        _cpu_command("scripts.paper_program_validate", "research/paper-1")
        _live_gate(run_name)
        entry = _qualification_runs()[run_name]
        summary = Path(entry["checkpoint_directory"]) / "run_summary.json"
        service = f"{_unit_stem(run_name)}-launch.service"
        _install_watcher(run_name, summary, service)
        environment = os.environ.copy()
        environment.pop("UV_PROJECT_ENVIRONMENT", None)
        environment.update(
            {"WANDB_MODE": "offline", "PYTHONUNBUFFERED": "1", "UV_NO_PROGRESS": "1"}
        )
        command = entry["launch"].split()
        command[0] = str(UV)
        os.execvpe(str(UV), command, environment)


def main(argv=None):
    args = arguments(argv)
    if args.command == "finalize":
        finalize(args.run, args.training_unit, args.trigger_unit)
    else:
        launch(args.run)


if __name__ == "__main__":
    main()
