"""Advance the fixed Paper 1 baseline sequence from successful file events."""

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import select
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "research/paper-1/experiment_program.json"
PLAN = ROOT / "research/paper-1/baseline_analysis.json"
AUTOMATION = ROOT / "research/paper-1/baseline_automation_v1.json"
STORAGE_REPORT = ROOT / "results/Speck-Paper1/baseline-storage-volume-qualified.json"
RESULTS = ROOT / "results/Speck-Paper1/runs"
TRANSITIONS = ROOT / "results/Speck-Paper1/transitions"
TARGET = ROOT / "results/Speck-Paper1/baseline-time-to-quality-target.json"
ANALYSIS = ROOT / "results/Speck-Paper1/baseline-analysis.json"
CPU_ENV = ROOT / ".venv-paper1-cpu"
UV = Path("/home/alkin/.local/bin/uv")
LOCK = Path("/run/user") / str(os.getuid()) / "speck-paper1-baseline.lock"
VOLUME_UUID = "b64b59d1-ea2c-4206-9171-b7cd739f3eff"
GPU_UUID = "GPU-6e7f2d05-ac19-3812-ff41-33079fd67bfd"


def _runs():
    storage = load_object(STORAGE_REPORT)
    return {
        entry["run"]: entry for entry in storage["operational_binding"]["runs"]
    }


def _candidate_runs():
    return [
        run for run in _runs() if run.endswith("five_cache_kda_gqa")
    ]


def _dense_pair_2():
    return next(
        run
        for run in _runs()
        if "pair-2-seed-44-order-1073741824-dense_global_param_match" in run
    )


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
        raise ValueError("automatic baseline continuation requires a clean repository")


def validate_automation_contract():
    contract = load_object(AUTOMATION)
    if (
        contract.get("format") != "speck_paper_baseline_automation_contract"
        or contract.get("format_version") != 1
        or contract.get("status") != "frozen_before_dense_control_2_completion"
        or contract.get("implementation", {}).get("runner_sha256")
        != file_sha256(__file__)
        or contract.get("inputs", {}).get("analysis_plan_sha256")
        != file_sha256(PLAN)
        or contract.get("inputs", {}).get("storage_qualification_sha256")
        != file_sha256(STORAGE_REPORT)
    ):
        raise ValueError("baseline automation contract does not match its frozen inputs")
    return contract


def _cpu_command(*arguments):
    environment = os.environ.copy()
    environment["UV_PROJECT_ENVIRONMENT"] = str(CPU_ENV)
    environment["UV_NO_PROGRESS"] = "1"
    return run(
        [str(UV), "run", "--extra", "cpu", "python", "-m", *arguments],
        env=environment,
    )


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


def _result_reference(path, pair):
    report = load_object(path)
    if (
        report.get("format") != "speck_paper_baseline_run_result"
        or report.get("status") != "complete_qualified"
        or report.get("pair", {}).get("pair") != pair
        or report.get("training_tokens") != 131_072_000
        or report.get("non_finite_steps") != 0
    ):
        raise ValueError(f"collected result is not qualified: {path}")
    return {
        "pair": pair,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": file_sha256(path),
        "status": report["status"],
    }


def _collect(run_name, entry):
    output = _result_path(run_name)
    if output.exists():
        raise FileExistsError(f"result already exists: {output}")
    _cpu_command(
        "scripts.paper_baseline_analyze",
        "collect",
        str(PLAN.relative_to(ROOT)),
        entry["experiment"],
        "--checkpoint-dir",
        entry["checkpoint_directory"],
        "--output",
        str(output.relative_to(ROOT)),
    )
    return output


def _lock_target(control_entries):
    if TARGET.exists():
        raise FileExistsError(f"target lock already exists: {TARGET}")
    paths = [entry["path"] for entry in control_entries]
    _cpu_command(
        "scripts.paper_baseline_analyze",
        "lock-target",
        str(PLAN.relative_to(ROOT)),
        *paths,
        "--output",
        str(TARGET.relative_to(ROOT)),
    )
    target = load_object(TARGET)
    control_losses = [
        load_object(ROOT / entry["path"])["final_validation"]["validation_loss"]
        for entry in control_entries
    ]
    expected = math.ceil(max(control_losses) * 1_000_000) / 1_000_000
    if (
        target.get("status") != "locked_from_controls_before_candidate_analysis"
        or target.get("validation_loss_target") != expected
    ):
        raise ValueError("control-only target lock is invalid")
    return {
        "path": TARGET.relative_to(ROOT).as_posix(),
        "sha256": file_sha256(TARGET),
        "status": target["status"],
    }


def _analyze(control_entries, candidate_entries, target_reference):
    if ANALYSIS.exists():
        raise FileExistsError(f"proxy analysis already exists: {ANALYSIS}")
    paths = [entry["path"] for entry in control_entries + candidate_entries]
    _cpu_command(
        "scripts.paper_baseline_analyze",
        "analyze",
        str(PLAN.relative_to(ROOT)),
        *paths,
        "--target-lock",
        target_reference["path"],
        "--output",
        str(ANALYSIS.relative_to(ROOT)),
    )
    report = load_object(ANALYSIS)
    if report.get("status") != "complete_proxy_evidence_no_promotion_authority":
        raise ValueError("completed proxy analysis is invalid")
    return {
        "path": ANALYSIS.relative_to(ROOT).as_posix(),
        "sha256": file_sha256(ANALYSIS),
        "status": report["status"],
    }


def _commit(paths, message):
    run(["git", "add", *[str(Path(path).relative_to(ROOT)) for path in paths]])
    run(["git", "commit", "-m", message])


def _schedule(run_name, delay="15m"):
    pair = _candidate_runs().index(run_name)
    unit = f"speck-paper1-baseline-candidate-pair{pair}-launch"
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
    result = run(command)
    print(result.stdout.strip(), flush=True)


def finalize(run_name, training_unit, trigger_unit):
    allowed = {_dense_pair_2(), *_candidate_runs()}
    if run_name not in allowed:
        raise ValueError("run is outside the automatic continuation sequence")
    _stop_trigger(trigger_unit)
    _wait_for_process_exit(training_unit)
    with LOCK.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        validate_automation_contract()
        git_clean()
        runs = _runs()
        entry = runs[run_name]
        result_path = _collect(run_name, entry)
        program = load_object(PROGRAM)
        evidence = program["baseline_evidence"]
        changed = [result_path, PROGRAM]
        next_run = None
        if run_name == _dense_pair_2():
            if len(evidence["dense_control_results"]) != 2:
                raise ValueError("dense-control sequence is not ready for pair 2")
            if any(
                Path(runs[candidate]["checkpoint_directory"]).exists()
                or _result_path(candidate).exists()
                for candidate in _candidate_runs()
            ):
                raise ValueError("candidate output exists before the control-only target lock")
            evidence["dense_control_results"].append(_result_reference(result_path, 2))
            evidence["time_to_quality_target"] = _lock_target(
                evidence["dense_control_results"]
            )
            changed.append(TARGET)
            next_run = _candidate_runs()[0]
            message = "Lock Paper 1 control target"
        else:
            pair = _candidate_runs().index(run_name)
            candidates = evidence.setdefault("candidate_results", [])
            if len(evidence["dense_control_results"]) != 3 or len(candidates) != pair:
                raise ValueError("candidate sequence is not ready for this result")
            candidates.append(_result_reference(result_path, pair))
            if pair < 2:
                next_run = _candidate_runs()[pair + 1]
                message = f"Register Paper 1 candidate pair {pair}"
            else:
                evidence["proxy_analysis_result"] = _analyze(
                    evidence["dense_control_results"],
                    candidates,
                    evidence["time_to_quality_target"],
                )
                changed.append(ANALYSIS)
                message = "Complete Paper 1 paired proxy analysis"
        atomic_json(PROGRAM, program)
        transition = {
            "format": "speck_paper_baseline_automatic_transition",
            "format_version": 1,
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
        raise ValueError("automatic launch requires exactly one visible GPU")
    name, uuid, total_mib, used_mib, utilization, temperature = rows[0]
    processes = run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ]
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
        raise ValueError(f"automatic launch GPU gate failed: {state}")
    return state


def _install_watcher(run_name, summary_path, training_unit):
    pair = _candidate_runs().index(run_name)
    unit = f"speck-paper1-baseline-candidate-pair{pair}-finalize"
    command = [
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
    result = run(command)
    print(result.stdout.strip(), flush=True)


def launch(run_name):
    candidates = _candidate_runs()
    if run_name not in candidates:
        raise ValueError("only candidate runs may be launched automatically")
    with LOCK.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        validate_automation_contract()
        git_clean()
        program = load_object(PROGRAM)
        evidence = program["baseline_evidence"]
        pair = candidates.index(run_name)
        if (
            len(evidence["dense_control_results"]) != 3
            or "time_to_quality_target" not in evidence
            or len(evidence.get("candidate_results", ())) != pair
        ):
            raise ValueError("program evidence is not ready for automatic candidate launch")
        entry = _runs()[run_name]
        checkpoint = Path(entry["checkpoint_directory"])
        if checkpoint.exists() or _result_path(run_name).exists():
            raise FileExistsError("candidate output already exists")
        if run(["systemctl", "--user", "is-active", "speck-helmet-download.service"], check=False).stdout.strip() == "active":
            raise ValueError("HELMET transfer must remain inactive during training")
        _gpu_state()
        mount = run(["findmnt", "-no", "UUID,FSTYPE,OPTIONS", "/mnt/speck-data"]).stdout
        if VOLUME_UUID not in mount or "ext4" not in mount:
            raise ValueError("checkpoint volume identity changed")
        if shutil.disk_usage("/mnt/speck-data").free < 17_179_869_184:
            raise ValueError("checkpoint volume is below the frozen proxy free-space floor")
        memory = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, raw = line.split(":", 1)
            memory[key] = int(raw.strip().split()[0]) * 1024
        if memory["MemAvailable"] < 12_884_901_888:
            raise ValueError("host available memory is below the frozen proxy launch floor")
        _cpu_command("scripts.paper_program_validate", "research/paper-1")
        summary = checkpoint / "run_summary.json"
        service = f"speck-paper1-baseline-candidate-pair{pair}-launch.service"
        _install_watcher(run_name, summary, service)
        environment = os.environ.copy()
        environment.pop("UV_PROJECT_ENVIRONMENT", None)
        environment.update(
            {"WANDB_MODE": "offline", "PYTHONUNBUFFERED": "1", "UV_NO_PROGRESS": "1"}
        )
        arguments = entry["launch"].split()
        arguments[0] = str(UV)
        os.execvpe(str(UV), arguments, environment)


def main(argv=None):
    args = arguments(argv)
    if args.command == "finalize":
        finalize(args.run, args.training_unit, args.trigger_unit)
    else:
        launch(args.run)


if __name__ == "__main__":
    main()
