"""Bind and supervise finite R0 attempts with preserved failures and conservative budget reservations."""

import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from speck.provenance.io import durable_json, file_sha256
from speck.provenance.readiness import relocated_source_path
from speck.provenance.repository import repository_root


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read_bound(root, binding):
    path = root / binding["path"]
    if file_sha256(path) != binding["sha256"]:
        raise ValueError(f"R0 input hash differs: {binding['path']}")
    return json.loads(path.read_text())


def validate_settings(settings):
    limits = {
        "device_batch_size": (1, 8),
        "accumulation": (1, 16),
        "warmup_steps": (1, 20),
        "measured_steps": (1, 100),
        "timeout_seconds": (1, 3600),
        "kill_grace_seconds": (1, 30),
        "collective_timeout_seconds": (1, 300),
        "seed": (0, 2**31 - 1),
        "minimum_free_disk_bytes": (1, 2**50),
    }
    required = set(limits) | {
        "lr",
        "weight_decay",
        "grad_clip",
        "compile",
        "activation_checkpointing",
        "loss_backend",
        "resume_tolerance",
    }
    if set(settings) != required:
        raise ValueError("R0 execution settings are missing or unknown")
    for key, (low, high) in limits.items():
        value = settings[key]
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"invalid bounded R0 setting: {key}")
    for key in ("lr", "weight_decay", "grad_clip"):
        value = settings[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"invalid numerical R0 setting: {key}")
    for key in ("compile", "activation_checkpointing"):
        if type(settings[key]) is not bool:
            raise ValueError(f"invalid boolean R0 setting: {key}")
    if settings["loss_backend"] not in ("torch", "liger"):
        raise ValueError("unsupported R0 loss backend")
    tolerance = settings["resume_tolerance"]
    if set(tolerance) != {"rtol", "atol"} or any(
        type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1e-4
        for v in tolerance.values()
    ):
        raise ValueError("invalid bounded R0 resume tolerance")


def prepare_request(plan_path, case_id, workers, allocated_gpus, root=None):
    root = Path(root or repository_root()).resolve()
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    if (plan.get("format"), plan.get("format_version"), plan.get("training_authority")) != (
        "speck_r0_execution_preparation",
        1,
        False,
    ):
        raise ValueError("unsupported R0 execution preparation plan")
    validate_settings(plan["settings"])
    if (
        type(workers) is not int
        or workers not in (1, 4)
        or type(allocated_gpus) is not int
        or not workers <= allocated_gpus <= 4
    ):
        raise ValueError("R0 workers must be one/four and cannot exceed allocated GPUs")
    shapes = read_bound(root, plan["shape_result"])
    shape_plan = read_bound(root, plan["shape_plan"])
    if shapes["plan"]["sha256"] != plan["shape_plan"]["sha256"]:
        raise ValueError("shape result and plan identities differ")
    catalog = json.loads((root / "research/catalog.json").read_text())
    values = {}
    for role, binding in shape_plan["inputs"].items():
        values[role] = read_bound(root, binding)
        if role != "retained_geometry" and catalog["active_contracts"][role] != binding:
            # Catalog entries can carry additional metadata; identity is path and SHA only.
            selected = catalog["active_contracts"][role]
            if any(selected[k] != binding[k] for k in ("path", "sha256")):
                raise ValueError("R0 input no longer selected by catalog")
        if shapes["inputs"][role]["sha256"] != binding["sha256"]:
            raise ValueError("shape evidence differs from selected input")
    for row in shapes["implementation"]:
        path = relocated_source_path(
            row["path"], shapes["plan"]["path"], plan["shape_plan"]["path"]
        )
        if file_sha256(root / path) != row["sha256"]:
            raise ValueError("shape implementation changed; successor qualification required")
    r0 = next(p for p in values["execution"]["phases"] if p["id"] == "R0")
    if plan["r0_gpu_hour_ceiling"] != r0["gpu_hours"] or r0["gpu_hours"] != 70:
        raise ValueError("R0 allocation differs")
    case = next((c for c in shapes["cases"] if c["id"] == case_id), None)
    if case is None:
        raise ValueError("unknown checked R0 case")
    implementation_paths = sorted(
        {
            "speck/operations/r0_executor.py",
            "speck/operations/r0_worker.py",
            "scripts/r0_execute.py",
            "speck/training/step.py",
            "speck/training/checkpoint.py",
            "speck/provenance/io.py",
            "speck/provenance/readiness.py",
            "speck/provenance/repository.py",
            "uv.lock",
            *[
                str(
                    relocated_source_path(
                        r["path"], shapes["plan"]["path"], plan["shape_plan"]["path"]
                    )
                )
                for r in shapes["implementation"]
            ],
        }
    )
    request = {
        "format": "speck_r0_bounded_attempt_request",
        "format_version": 1,
        "plan": {"path": str(plan_path), "sha256": file_sha256(plan_path)},
        "shape_result": plan["shape_result"],
        "case": case,
        "settings": plan["settings"],
        "world_size": workers,
        "allocated_gpus": allocated_gpus,
        "r0_gpu_hour_ceiling": plan["r0_gpu_hour_ceiling"],
        "implementation": [
            {"path": p, "sha256": file_sha256(root / p)} for p in implementation_paths
        ],
        "input_policy": "CPU torch.Generator(seed + microbatch_ordinal * world_size + rank); int64 uniform [0,32000), length+1; payload hashes recorded per step. No source text, tokenizer sampling or role IDs. Synthetic repetition is not a corpus exposure policy.",
        "precision_policy": "FP32 parameter/optimizer storage with the existing model's BF16 CUDA activations; no blanket BF16 conversion of optimizer/recurrent state.",
        "scientific_training_authority": False,
    }
    request["request_sha256"] = fingerprint(request)
    return request


def supervise(command, directory, timeout_seconds, grace_seconds, worker_count=1):
    """Own each rank's process group directly; torchrun ranks create independent sessions."""
    started = time.monotonic()
    processes = []
    reason, error = "exited", None
    previous_term = signal.getsignal(signal.SIGTERM)

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    def send_all(sig):
        for process in processes:
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, interrupted)
    try:
        with (directory / "worker.log").open("xb") as log:
            for rank in range(worker_count):
                environment = {
                    **os.environ,
                    "RANK": str(rank),
                    "LOCAL_RANK": str(rank),
                    "WORLD_SIZE": str(worker_count),
                }
                processes.append(
                    subprocess.Popen(
                        command,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                )
                durable_json(
                    directory / "processes.json",
                    {
                        "supervisor_pid": os.getpid(),
                        "ranks": [
                            {"rank": i, "pid": p.pid, "process_group": p.pid}
                            for i, p in enumerate(processes)
                        ],
                        "boundary": "Diagnostic process IDs only; check current owner, command, allocation and process start time before recovery because IDs can be reused. A killed supervisor leaves its reservation unresolved.",
                    },
                )
            while True:
                codes = [p.poll() for p in processes]
                if any(c is not None and c != 0 for c in codes):
                    reason = "worker_failure"
                    break
                if all(c is not None for c in codes):
                    break
                remaining = timeout_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    reason = "timeout"
                    break
                time.sleep(min(0.05, remaining))
    except KeyboardInterrupt:
        reason = "interrupted"
    except Exception as exception:
        reason, error = "launch_failure", repr(exception)
    finally:
        # Terminate every known rank group, including descendants whose rank leader already exited.
        send_all(signal.SIGTERM)
        deadline = time.monotonic() + grace_seconds
        for process in processes:
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        send_all(signal.SIGKILL)
        for process in processes:
            process.wait()
        signal.signal(signal.SIGTERM, previous_term)
    codes = [p.returncode for p in processes]
    return {
        "termination": reason,
        "returncode": next((c for c in codes if c != 0), 0) if len(codes) == worker_count else None,
        "rank_returncodes": codes,
        "supervised_wall_seconds": time.monotonic() - started,
        "launch_error": error,
    }


def summarize_attempt(request, directory, execution):
    ranks = []
    for rank in range(request["world_size"]):
        path = directory / f"rank-{rank}-result.json"
        if path.exists():
            report = json.loads(path.read_text())
            if (report["rank"], report["world_size"], report["request_sha256"]) != (
                rank,
                request["world_size"],
                request["request_sha256"],
            ):
                raise ValueError("rank result identity differs")
            ranks.append(
                {
                    "rank": rank,
                    "status": report["status"],
                    "path": path.name,
                    "sha256": file_sha256(path),
                }
            )
    passed = (
        execution["termination"] == "exited"
        and execution["returncode"] == 0
        and len(ranks) == request["world_size"]
        and all(r["status"] == "bounded_synthetic_checks_pass" for r in ranks)
    )
    return {
        "format": "speck_r0_supervised_attempt",
        "format_version": 1,
        "status": "bounded_synthetic_checks_pass" if passed else "attempt_failed_or_incomplete",
        "request_sha256": request["request_sha256"],
        "execution": execution,
        "ranks": ranks,
        "observed_allocated_gpu_hours": execution["supervised_wall_seconds"]
        * request["allocated_gpus"]
        / 3600,
        "scheduler_total_gpu_hours": None,
        "boundary": "Observed hours cover supervisor launch through worker-group termination, charging all declared allocated GPUs even in single-worker mode. Scheduler startup/idle/cleanup outside this window and other R0 work need separate accounting. A pass is bounded synthetic execution, not complete R0 or model-launch authority.",
    }


def run_attempt(request, ledger_directory, prior_gpu_hours):
    """One locked phase ledger; reserve worst-case time without silently refunding unused headroom."""
    if (
        type(prior_gpu_hours) not in (int, float)
        or not math.isfinite(prior_gpu_hours)
        or prior_gpu_hours < 0
    ):
        raise ValueError("prior R0 hours must be explicitly accounted and nonnegative")
    directory = Path(ledger_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger_path = directory / "ledger.json"
        ledger = (
            json.loads(ledger_path.read_text())
            if ledger_path.exists()
            else {
                "format": "speck_r0_reservation_ledger",
                "format_version": 1,
                "ceiling_gpu_hours": request["r0_gpu_hour_ceiling"],
                "external_prior_gpu_hours": prior_gpu_hours,
                "attempts": [],
            }
        )
        if (
            ledger["external_prior_gpu_hours"] != prior_gpu_hours
            or ledger["ceiling_gpu_hours"] != request["r0_gpu_hour_ceiling"]
        ):
            raise ValueError("ledger accounting changed; reconcile explicitly before execution")
        if any(a["state"] == "reserved_or_running" for a in ledger["attempts"]):
            raise ValueError(
                "unresolved prior attempt; preserve and reconcile process/scheduler state first"
            )
        for previous in ledger["attempts"]:
            reserved = previous["reserved_gpu_hours"]
            observed = previous["observed_allocated_gpu_hours"]
            if (
                any(
                    type(v) not in (int, float) or not math.isfinite(v) or v < 0
                    for v in (reserved, observed)
                )
                or reserved < observed
            ):
                raise ValueError("invalid previous R0 accounting")
            result_path = directory / previous["id"] / "result.json"
            if file_sha256(result_path) != previous["result_sha256"]:
                raise ValueError("previous R0 result changed; reconcile before further execution")
        settings = request["settings"]
        reservation = (
            (settings["timeout_seconds"] + settings["kill_grace_seconds"])
            * request["allocated_gpus"]
            / 3600
        )
        if (
            prior_gpu_hours + sum(a["reserved_gpu_hours"] for a in ledger["attempts"]) + reservation
            > ledger["ceiling_gpu_hours"]
        ):
            raise ValueError("R0 conservative reservation would exceed remaining ceiling")
        if shutil.disk_usage(directory).free < settings["minimum_free_disk_bytes"]:
            raise ValueError("R0 checkpoint storage floor not met")
        attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:12]
        attempt_dir = directory / attempt_id
        attempt_dir.mkdir()
        durable_json(attempt_dir / "request.json", request)
        row = {
            "id": attempt_id,
            "state": "reserved_or_running",
            "reserved_gpu_hours": reservation,
            "request_sha256": request["request_sha256"],
        }
        ledger["attempts"].append(row)
        durable_json(ledger_path, ledger)
        command = [
            sys.executable,
            "-m",
            "scripts.r0_execute",
            "--worker-request",
            str(attempt_dir / "request.json"),
        ]
        execution = supervise(
            command,
            attempt_dir,
            settings["timeout_seconds"],
            settings["kill_grace_seconds"],
            request["world_size"],
        )
        result = summarize_attempt(request, attempt_dir, execution)
        durable_json(attempt_dir / "result.json", result)
        row.update(
            state=result["status"],
            observed_allocated_gpu_hours=result["observed_allocated_gpu_hours"],
            result_sha256=file_sha256(attempt_dir / "result.json"),
        )
        # If OS termination itself exceeded the bound, preserve the larger debit rather than hiding it.
        row["reserved_gpu_hours"] = max(reservation, result["observed_allocated_gpu_hours"])
        durable_json(ledger_path, ledger)
        return {"attempt_directory": str(attempt_dir), **result}
