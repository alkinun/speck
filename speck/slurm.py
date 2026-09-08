"""Validate and operate immutable single-node Slurm wave manifests."""

import hashlib
import json
import os
import re
import shlex
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path

FORMAT = "speck_slurm_wave"
FORMAT_VERSION = 1
PLAN_FORMAT = "speck_flagship_execution_plan"
TOTAL_GPU_HOURS = 5_000
MANDATORY_GPU_HOURS = 4_111
RESERVE_GPU_HOURS = 889
REQUEUE_EXIT_CODE = 99

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,47}")
_SBATCH_VALUE = re.compile(r"[A-Za-z0-9_.:@/+,-]+")
_TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}
_ACTIVE_STATES = {
    "COMPLETING",
    "CONFIGURING",
    "PENDING",
    "REQUEUED",
    "REQUEUE_FED",
    "REQUEUE_HOLD",
    "RESIZING",
    "RUNNING",
    "SUSPENDED",
}
_RETRYABLE_STATES = {"BOOT_FAIL", "NODE_FAIL", "PREEMPTED", "REVOKED", "TIMEOUT"}


def sha256_file(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def manifest_digest(path):
    """Return the byte identity used to freeze one wave manifest."""

    return sha256_file(path)


def _exact_keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{name} must contain exactly: {', '.join(sorted(expected))}")


def _integer(value, name, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def _path(value, base, name):
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _identity(value, base, name):
    _exact_keys(value, {"path", "sha256"}, name)
    if not isinstance(value["sha256"], str) or not _SHA256.fullmatch(value["sha256"]):
        raise ValueError(f"{name} sha256 must be lowercase SHA-256")
    return {"path": str(_path(value["path"], base, name)), "sha256": value["sha256"]}


def _check_plan(plan):
    if plan.get("format") != PLAN_FORMAT or plan.get("budget") != {
        "gpu_hours": TOTAL_GPU_HOURS,
        "mandatory_gpu_hours": MANDATORY_GPU_HOURS,
        "reserve_gpu_hours": RESERVE_GPU_HOURS,
        "full_node_days": plan.get("budget", {}).get("full_node_days"),
    }:
        raise ValueError("execution plan does not preserve the 4,111 + 889 GPU-hour budget")
    phases = plan.get("phases")
    if not isinstance(phases, list):
        raise ValueError("execution plan phases are missing")
    mandatory = sum(item.get("gpu_hours", 0) for item in phases if not item.get("conditional"))
    reserve = sum(item.get("gpu_hours", 0) for item in phases if item.get("conditional"))
    if mandatory != MANDATORY_GPU_HOURS or reserve != RESERVE_GPU_HOURS:
        raise ValueError("execution plan phase accounting does not preserve mandatory and reserve")


def _validate_job(raw, base, phase_ids):
    _exact_keys(
        raw,
        {
            "id",
            "phase",
            "kind",
            "allocation",
            "resources",
            "array",
            "command",
            "working_directory",
            "identities",
            "max_retries",
            "depends_on",
        },
        "job",
    )
    if not isinstance(raw["id"], str) or not _ID.fullmatch(raw["id"]):
        raise ValueError("job id must use lowercase letters, digits, and hyphens")
    if raw["phase"] not in phase_ids or raw["phase"] in {"P0", "P8"}:
        raise ValueError(f"job {raw['id']} has a non-compute phase")
    if raw["kind"] not in {"train", "collect", "eval"}:
        raise ValueError(f"job {raw['id']} has an unsupported kind")
    if raw["allocation"] not in {"mandatory", "reserve"}:
        raise ValueError(f"job {raw['id']} has an unsupported allocation")
    if (raw["allocation"] == "reserve") != (raw["phase"] == "P7"):
        raise ValueError(f"job {raw['id']} must keep mandatory and P7 reserve phases separate")
    resources = raw["resources"]
    _exact_keys(
        resources,
        {"nodes", "gpus", "cpus_per_task", "memory_mb", "walltime_minutes", "signal_seconds"},
        f"job {raw['id']} resources",
    )
    if resources["nodes"] != 1 or resources["gpus"] not in {1, 4}:
        raise ValueError(f"job {raw['id']} must request one node and either one or four GPUs")
    _integer(resources["cpus_per_task"], "cpus_per_task", 1)
    _integer(resources["memory_mb"], "memory_mb", 1)
    _integer(resources["walltime_minutes"], "walltime_minutes", 1)
    signal_seconds = _integer(resources["signal_seconds"], "signal_seconds", 30)
    if signal_seconds >= resources["walltime_minutes"] * 60:
        raise ValueError(f"job {raw['id']} signal must precede timeout")
    array = raw["array"]
    if resources["gpus"] == 1:
        _exact_keys(array, {"indices", "max_parallel"}, f"job {raw['id']} array")
        indices = array["indices"]
        if (
            not isinstance(indices, list)
            or not indices
            or len(indices) != len(set(indices))
            or any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in indices
            )
        ):
            raise ValueError(f"job {raw['id']} array indices must be unique non-negative integers")
        if indices != sorted(indices):
            raise ValueError(f"job {raw['id']} array indices must be sorted")
        _integer(array["max_parallel"], "max_parallel", 1, 4)
        if array["max_parallel"] > len(indices):
            raise ValueError(f"job {raw['id']} array concurrency exceeds its task count")
    elif array is not None:
        raise ValueError(f"four-GPU job {raw['id']} cannot be an array")
    command = raw["command"]
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(part, str) or not part or "\x00" in part for part in command)
    ):
        raise ValueError(f"job {raw['id']} command must be a non-empty argv list")
    placeholders = sum(part.count("{array_index}") for part in command)
    if resources["gpus"] == 1 and placeholders < 1:
        raise ValueError(f"array job {raw['id']} command must use {{array_index}}")
    if resources["gpus"] == 4 and placeholders:
        raise ValueError(f"four-GPU job {raw['id']} cannot use {{array_index}}")
    _integer(raw["max_retries"], "max_retries", 0, 2)
    if raw["allocation"] == "reserve" and raw["max_retries"]:
        raise ValueError(f"reserve job {raw['id']} cannot automate retry spending")
    if raw["kind"] == "train" and (
        "--slurm-requeue-resume" not in command or "scripts.slurm_base_train" not in command
    ):
        raise ValueError(f"train job {raw['id']} must use the signal-safe Slurm trainer")
    dependencies = raw["depends_on"]
    if not isinstance(dependencies, list) or any(
        not isinstance(item, str) for item in dependencies
    ):
        raise ValueError(f"job {raw['id']} dependencies must be job ids")
    if dependencies and raw["kind"] not in {"collect", "eval"}:
        raise ValueError("dependencies are allowed only for mechanical collect/eval jobs")
    identities = raw["identities"]
    if not isinstance(identities, list) or not identities:
        raise ValueError(f"job {raw['id']} must bind config/data identities")
    normalized_identities = []
    roles = set()
    for number, identity in enumerate(identities):
        _exact_keys(identity, {"role", "path", "sha256"}, f"job {raw['id']} identity")
        if identity["role"] not in {"config", "data", "checkpoint", "authority", "tool"}:
            raise ValueError(f"job {raw['id']} identity {number} has an unsupported role")
        roles.add(identity["role"])
        bound = _identity(
            {"path": identity["path"], "sha256": identity["sha256"]},
            base,
            f"job {raw['id']} identity",
        )
        normalized_identities.append({"role": identity["role"], **bound})
    if not {"config", "data"}.issubset(roles):
        raise ValueError(f"job {raw['id']} must bind at least one config and one data identity")
    return {
        **raw,
        "working_directory": str(_path(raw["working_directory"], base, "working directory")),
        "identities": normalized_identities,
    }


def load_wave(path):
    """Load, strictly validate, and normalize one immutable wave manifest."""

    source = Path(path).expanduser().resolve()
    raw = json.loads(source.read_text(encoding="utf-8"))
    _exact_keys(
        raw,
        {
            "format",
            "format_version",
            "wave_id",
            "created_at_utc",
            "budget",
            "plan",
            "repository",
            "jobs",
        },
        "wave manifest",
    )
    if raw["format"] != FORMAT or raw["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported wave manifest")
    if not isinstance(raw["wave_id"], str) or not _ID.fullmatch(raw["wave_id"]):
        raise ValueError("wave_id must use lowercase letters, digits, and hyphens")
    try:
        created = datetime.fromisoformat(raw["created_at_utc"].replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise ValueError("created_at_utc must be an ISO-8601 timestamp") from error
    if created.tzinfo is None:
        raise ValueError("created_at_utc must include a timezone")
    if raw["budget"] != {
        "total_gpu_hours": TOTAL_GPU_HOURS,
        "mandatory_gpu_hours": MANDATORY_GPU_HOURS,
        "reserve_gpu_hours": RESERVE_GPU_HOURS,
    }:
        raise ValueError("wave budget must preserve protected 4,111 + 889 accounting")
    plan_identity = _identity(raw["plan"], source.parent, "execution plan")
    plan_path = Path(plan_identity["path"])
    if not plan_path.is_file() or sha256_file(plan_path) != plan_identity["sha256"]:
        raise ValueError("execution plan identity mismatch")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    _check_plan(plan)
    repository = raw["repository"]
    _exact_keys(repository, {"path", "commit", "require_clean"}, "repository")
    if not isinstance(repository["commit"], str) or not _COMMIT.fullmatch(repository["commit"]):
        raise ValueError("repository commit must be a full lowercase Git commit")
    if repository["require_clean"] is not True:
        raise ValueError("wave manifests must require a clean worktree")
    repository_path = _path(repository["path"], source.parent, "repository")
    jobs = raw["jobs"]
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("wave must contain at least one job")
    phase_ids = {phase["id"] for phase in plan["phases"]}
    normalized_jobs = [_validate_job(job, source.parent, phase_ids) for job in jobs]
    ids = [job["id"] for job in normalized_jobs]
    if len(ids) != len(set(ids)):
        raise ValueError("job ids must be unique")
    positions = {job_id: index for index, job_id in enumerate(ids)}
    for index, job in enumerate(normalized_jobs):
        if len(job["depends_on"]) != len(set(job["depends_on"])):
            raise ValueError(f"job {job['id']} dependencies must be unique")
        if any(
            parent not in positions or positions[parent] >= index for parent in job["depends_on"]
        ):
            raise ValueError(
                f"job {job['id']} dependencies must refer to earlier jobs in this wave"
            )
    planned = {"mandatory": 0.0, "reserve": 0.0}
    for job in normalized_jobs:
        task_count = len(job["array"]["indices"]) if job["array"] else 1
        maximum_attempts = 1 + job["max_retries"]
        hours = (
            job["resources"]["gpus"]
            * job["resources"]["walltime_minutes"]
            * task_count
            * maximum_attempts
            / 60
        )
        planned[job["allocation"]] += hours
    if planned["mandatory"] > MANDATORY_GPU_HOURS or planned["reserve"] > RESERVE_GPU_HOURS:
        raise ValueError("wave maximum-attempt GPU-hours exceed the protected budget pool")
    normalized = {
        **raw,
        "plan": plan_identity,
        "repository": {**repository, "path": str(repository_path)},
        "jobs": normalized_jobs,
    }
    return normalized, manifest_digest(source), source, planned


def _git(repository, *args):
    return subprocess.run(
        ["git", *args], cwd=repository, check=True, capture_output=True, text=True
    ).stdout.strip()


def preflight_wave(manifest):
    """Verify Git and every config/data identity immediately before rendering/submission."""

    repository = Path(manifest["repository"]["path"])
    if _git(repository, "rev-parse", "HEAD") != manifest["repository"]["commit"]:
        raise ValueError("wave Git commit is not checked out")
    if _git(repository, "status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("wave requires a clean Git worktree")
    verified = []
    for job in manifest["jobs"]:
        workdir = Path(job["working_directory"])
        if not workdir.is_dir():
            raise ValueError(f"job {job['id']} working directory is missing")
        for identity in job["identities"]:
            path = Path(identity["path"])
            if not path.is_file() or sha256_file(path) != identity["sha256"]:
                raise ValueError(f"job {job['id']} {identity['role']} identity mismatch")
            verified.append({"job_id": job["id"], **identity})
    return {
        "git_commit": manifest["repository"]["commit"],
        "identities_verified": len(verified),
        "verified": verified,
    }


def _walltime(minutes):
    days, remainder = divmod(minutes, 24 * 60)
    hours, minutes = divmod(remainder, 60)
    return f"{days}-{hours:02d}:{minutes:02d}:00" if days else f"{hours:02d}:{minutes:02d}:00"


def _array_spec(array):
    return ",".join(str(index) for index in array["indices"]) + f"%{array['max_parallel']}"


def _shell_command(command, array):
    parts = []
    for part in command:
        if array and part == "{array_index}":
            parts.append('"${SLURM_ARRAY_TASK_ID}"')
        elif "{array_index}" in part:
            raise ValueError("{array_index} must occupy a complete command argument")
        else:
            parts.append(shlex.quote(part))
    return " ".join(parts)


def render_job_script(job, digest, runtime_root, *, account=None, partition=None):
    """Render one script without inventing cluster account, partition, or GPU model values."""

    for name, value in (("account", account), ("partition", partition)):
        if value is not None and (not isinstance(value, str) or not _SBATCH_VALUE.fullmatch(value)):
            raise ValueError(f"invalid Slurm {name}")
    resources = job["resources"]
    runtime_root = Path(runtime_root).expanduser().resolve()
    log_dir = runtime_root / "logs" / digest / job["id"]
    signal_dir = runtime_root / "signals" / digest / job["id"]
    attempt_dir = runtime_root / "attempts" / digest / job["id"]
    log_token = "%A_%a" if job["array"] else "%j"
    directives = [
        f"#SBATCH --job-name={job['id']}-{digest[:8]}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={resources['cpus_per_task']}",
        f"#SBATCH --mem={resources['memory_mb']}M",
        f"#SBATCH --gres=gpu:{resources['gpus']}",
        f"#SBATCH --time={_walltime(resources['walltime_minutes'])}",
        f"#SBATCH --signal=B:USR1@{resources['signal_seconds']}",
        f"#SBATCH --output={log_dir}/{log_token}.out",
        f"#SBATCH --error={log_dir}/{log_token}.err",
    ]
    if job["array"]:
        directives.append(f"#SBATCH --array={_array_spec(job['array'])}")
    if account is not None:
        directives.append(f"#SBATCH --account={account}")
    if partition is not None:
        directives.append(f"#SBATCH --partition={partition}")
    command = _shell_command(job["command"], job["array"])
    retries = job["max_retries"]
    return (
        "#!/usr/bin/env bash\n"
        + "\n".join(directives)
        + "\nset -euo pipefail\n"
        + f"readonly SPECK_MANIFEST_SHA256={shlex.quote(digest)}\n"
        + f"readonly signal_dir={shlex.quote(str(signal_dir))}\n"
        + f"readonly attempt_dir={shlex.quote(str(attempt_dir))}\n"
        + 'readonly task_id="${SLURM_ARRAY_TASK_ID:-single}"\n'
        + 'readonly SPECK_REQUEUE_SIGNAL_FILE="${signal_dir}/${SLURM_JOB_ID:?}-'
        + '${task_id}"\n'
        + 'readonly attempt_file="${attempt_dir}/${SLURM_JOB_ID}-${task_id}"\n'
        + "attempt=${SPECK_RETRY_OFFSET:-0}\n"
        + 'if [[ -f "${attempt_file}" ]]; then read -r attempt < "${attempt_file}"; fi\n'
        + 'if [[ ! "${attempt}" =~ ^[0-9]+$ ]]; then echo "invalid retry state" >&2; exit 2; fi\n'
        + f'readonly SPECK_RUN_ID="{job["id"]}-{digest[:12]}-${{task_id}}-a${{attempt}}"\n'
        + "export SPECK_MANIFEST_SHA256 SPECK_RUN_ID SPECK_REQUEUE_SIGNAL_FILE\n"
        + 'export SPECK_RETRY_OFFSET="${attempt}"\n'
        + 'rm -f "${SPECK_REQUEUE_SIGNAL_FILE}"\n'
        + f"cd {shlex.quote(job['working_directory'])}\n"
        + "request_requeue() {\n"
        + '  : > "${SPECK_REQUEUE_SIGNAL_FILE}"\n'
        + "}\n"
        + "trap request_requeue USR1\n"
        + f"{command} &\n"
        + "child_pid=$!\n"
        + "set +e\n"
        + 'wait "${child_pid}"\n'
        + "status=$?\n"
        + 'while kill -0 "${child_pid}" 2>/dev/null; do\n'
        + '  wait "${child_pid}"\n'
        + "  status=$?\n"
        + "done\n"
        + "set -e\n"
        + f"if [[ $status -eq {REQUEUE_EXIT_CODE} ]]; then\n"
        + f"  if (( attempt < {retries} )); then\n"
        + "    next_attempt=$(( attempt + 1 ))\n"
        + '    temporary="${attempt_file}.tmp.$$"\n'
        + '    printf "%s\\n" "${next_attempt}" > "${temporary}"\n'
        + '    mv "${temporary}" "${attempt_file}"\n'
        + '    scontrol requeue "${SLURM_JOB_ID:?}"\n'
        + "    exit 0\n"
        + "  fi\n"
        + "fi\n"
        + 'exit "$status"\n'
    )


def freeze_wave(manifest, digest, source, runtime_root):
    destination = Path(runtime_root).expanduser().resolve() / "manifests" / f"{digest}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = source.read_bytes()
    if destination.exists():
        if destination.read_bytes() != payload:
            raise ValueError("frozen manifest digest collision")
    else:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    return destination


def render_wave(path, runtime_root, *, account=None, partition=None):
    manifest, digest, source, planned = load_wave(path)
    preflight = preflight_wave(manifest)
    frozen = freeze_wave(manifest, digest, source, runtime_root)
    scripts = {}
    script_hashes = {}
    site_payload = json.dumps(
        {"account": account, "partition": partition}, sort_keys=True, separators=(",", ":")
    )
    site_id = hashlib.sha256(site_payload.encode()).hexdigest()[:12]
    script_dir = Path(runtime_root).expanduser().resolve() / "scripts" / digest / site_id
    script_dir.mkdir(parents=True, exist_ok=True)
    for job in manifest["jobs"]:
        log_dir = Path(runtime_root).expanduser().resolve() / "logs" / digest / job["id"]
        log_dir.mkdir(parents=True, exist_ok=True)
        signal_dir = Path(runtime_root).expanduser().resolve() / "signals" / digest / job["id"]
        signal_dir.mkdir(parents=True, exist_ok=True)
        attempt_dir = Path(runtime_root).expanduser().resolve() / "attempts" / digest / job["id"]
        attempt_dir.mkdir(parents=True, exist_ok=True)
        script = script_dir / f"{job['id']}.sbatch"
        content = render_job_script(job, digest, runtime_root, account=account, partition=partition)
        if script.exists():
            if script.read_text(encoding="utf-8") != content:
                raise ValueError(
                    "rendered script changed for the same manifest and cluster options"
                )
        else:
            descriptor = os.open(script, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        scripts[job["id"]] = str(script)
        script_hashes[job["id"]] = sha256_file(script)
    return {
        "manifest_sha256": digest,
        "frozen_manifest": str(frozen),
        "rendering_id": site_id,
        "scripts": scripts,
        "script_sha256": script_hashes,
        "render_options": {"account": account, "partition": partition},
        "planned_gpu_hours": planned,
        "preflight": preflight,
    }


def _write_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _budget_commitments(runtime_root):
    committed = {"mandatory": 0.0, "reserve": 0.0}
    commitments = Path(runtime_root).expanduser().resolve() / "commitments"
    for path in commitments.glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        for allocation in committed:
            committed[allocation] += record["gpu_hours"].get(allocation, 0)
    return committed


def _reserve_commitment(runtime_root, digest, planned):
    root = Path(runtime_root).expanduser().resolve()
    lock = root / ".budget-lock"
    root.mkdir(parents=True, exist_ok=True)
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise RuntimeError(
            "another budget operation is active; inspect the runtime lock"
        ) from error
    try:
        if (root / "commitments" / f"{digest}.json").exists():
            raise ValueError("this immutable wave has already been submitted or budget-committed")
        committed = _budget_commitments(root)
        if (
            committed["mandatory"] + planned["mandatory"] > MANDATORY_GPU_HOURS
            or committed["reserve"] + planned["reserve"] > RESERVE_GPU_HOURS
        ):
            raise ValueError("submission would exceed a protected GPU-hour commitment pool")
        _write_exclusive(
            root / "commitments" / f"{digest}.json",
            {
                "format": "speck_slurm_budget_commitment",
                "format_version": 1,
                "manifest_sha256": digest,
                "committed_at_utc": datetime.now(timezone.utc).isoformat(),
                "gpu_hours": planned,
            },
        )
    finally:
        lock.rmdir()


def submit_wave(path, runtime_root, *, account=None, partition=None, runner=subprocess.run):
    """Submit mandatory jobs only; scientific promotion and reserve submission are not implemented."""

    manifest, digest, _, _ = load_wave(path)
    if any(job["allocation"] == "reserve" for job in manifest["jobs"]):
        raise ValueError(
            "reserve jobs require a separate human-operated launch; automatic spend refused"
        )
    rendered = render_wave(path, runtime_root, account=account, partition=partition)
    _reserve_commitment(runtime_root, digest, rendered["planned_gpu_hours"])
    submitted = {}
    events = []
    for sequence, job in enumerate(manifest["jobs"]):
        command = ["sbatch", "--parsable"]
        if job["depends_on"]:
            parents = [submitted[parent] for parent in job["depends_on"]]
            command.append(f"--dependency=afterok:{':'.join(parents)}")
        command.append(rendered["scripts"][job["id"]])
        result = runner(command, check=True, capture_output=True, text=True)
        job_id = result.stdout.strip().split(";", 1)[0]
        if not job_id.isdigit():
            raise ValueError("sbatch did not return a numeric job id")
        submitted[job["id"]] = job_id
        event_path = (
            Path(runtime_root).expanduser().resolve()
            / "submission-events"
            / digest
            / f"{sequence:04d}-{job['id']}.json"
        )
        _write_exclusive(
            event_path,
            {
                "format": "speck_slurm_submission_event",
                "format_version": 1,
                "manifest_sha256": digest,
                "logical_job_id": job["id"],
                "scheduler_job_id": job_id,
                "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        events.append(str(event_path))
    record = {
        "format": "speck_slurm_submission",
        "format_version": 1,
        "manifest_sha256": digest,
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "jobs": submitted,
        "allocations": {job["id"]: job["allocation"] for job in manifest["jobs"]},
        "attempt_offsets": {job["id"]: 0 for job in manifest["jobs"]},
        "budget_commitment_gpu_hours": rendered["planned_gpu_hours"],
        "submission_events": events,
        "rendering_id": rendered["rendering_id"],
        "render_options": rendered["render_options"],
        "script_sha256": rendered["script_sha256"],
        "reserve_spend_automated": False,
        "scientific_promotion_automated": False,
    }
    record_path = (
        Path(runtime_root).expanduser().resolve()
        / "submissions"
        / digest
        / f"{record['submitted_at_utc'].replace(':', '')}.json"
    )
    _write_exclusive(record_path, record)
    return record, record_path


def register_manual_reserve(
    path, scheduler_jobs, authorization_path, authorization_sha256, runtime_root
):
    """Record, but never submit, a human-authorized reserve wave for accounting."""

    manifest, digest, source, planned = load_wave(path)
    if any(job["allocation"] != "reserve" for job in manifest["jobs"]):
        raise ValueError("manual reserve registration requires an all-reserve wave")
    expected = {job["id"] for job in manifest["jobs"]}
    if (
        not isinstance(scheduler_jobs, dict)
        or set(scheduler_jobs) != expected
        or any(
            not isinstance(job_id, str) or not job_id.isdigit()
            for job_id in scheduler_jobs.values()
        )
    ):
        raise ValueError("manual scheduler ids must cover every reserve job exactly")
    if not isinstance(authorization_sha256, str) or not _SHA256.fullmatch(authorization_sha256):
        raise ValueError("reserve authorization sha256 must be lowercase SHA-256")
    authorization = Path(authorization_path).expanduser().resolve()
    if not authorization.is_file() or sha256_file(authorization) != authorization_sha256:
        raise ValueError("reserve authorization identity mismatch")
    required_authorities = {
        identity["path"]: identity["sha256"]
        for job in manifest["jobs"]
        for identity in job["identities"]
        if identity["role"] == "authority"
    }
    if required_authorities.get(str(authorization)) != authorization_sha256:
        raise ValueError("reserve authorization must be bound in every wave job")
    if any(
        not any(
            identity["role"] == "authority"
            and identity["path"] == str(authorization)
            and identity["sha256"] == authorization_sha256
            for identity in job["identities"]
        )
        for job in manifest["jobs"]
    ):
        raise ValueError("reserve authorization must be bound in every wave job")
    preflight_wave(manifest)
    frozen = freeze_wave(manifest, digest, source, runtime_root)
    _reserve_commitment(runtime_root, digest, planned)
    submitted_at = datetime.now(timezone.utc).isoformat()
    record = {
        "format": "speck_slurm_submission",
        "format_version": 1,
        "manifest_sha256": digest,
        "submitted_at_utc": submitted_at,
        "jobs": scheduler_jobs,
        "allocations": {job_id: "reserve" for job_id in expected},
        "attempt_offsets": {job_id: 0 for job_id in expected},
        "budget_commitment_gpu_hours": planned,
        "frozen_manifest": str(frozen),
        "manual_reserve_authorization": {
            "path": str(authorization),
            "sha256": authorization_sha256,
        },
        "reserve_spend_automated": False,
        "scientific_promotion_automated": False,
    }
    record_path = (
        Path(runtime_root).expanduser().resolve()
        / "submissions"
        / digest
        / f"{submitted_at.replace(':', '')}.json"
    )
    _write_exclusive(record_path, record)
    return record, record_path


def classify_state(state, exit_code="0:0"):
    normalized = state.split("+", 1)[0].split(maxsplit=1)[0].upper()
    if normalized == "COMPLETED" and exit_code == "0:0":
        return "success"
    if normalized in _ACTIVE_STATES:
        return "active"
    if normalized in _RETRYABLE_STATES:
        return "mechanical_retryable"
    if normalized == "OUT_OF_MEMORY":
        return "resource_configuration_failure"
    if normalized == "CANCELLED":
        return "operator_cancelled"
    if normalized in _TERMINAL_STATES or normalized == "COMPLETED":
        return "application_failure"
    return "unknown"


def _allocated_gpus(alloc_tres):
    matches = re.findall(r"(?:^|,)gres/gpu(?::[^,=]+)?=(\d+)(?:,|$)", alloc_tres)
    if not matches:
        matches = re.findall(r"(?:^|,)gpu=(\d+)(?:,|$)", alloc_tres)
    return int(matches[-1]) if matches else 0


def _sacct_time(value):
    return None if value in {"", "Unknown", "N/A", "None"} else value


def parse_sacct(text, expected_jobs):
    """Parse pipe-delimited sacct output while excluding duplicate batch/extern steps."""

    scheduler_to_logical = {scheduler: logical for logical, scheduler in expected_jobs.items()}
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = (line[:-1] if line.endswith("|") else line).split("|")
        if len(fields) != 10:
            raise ValueError("unexpected sacct field count")
        (
            job_id,
            name,
            state,
            exit_code,
            elapsed,
            alloc_tres,
            restart_count,
            submitted,
            started,
            ended,
        ) = fields
        root = job_id.split("_", 1)[0].split(".", 1)[0]
        if root not in scheduler_to_logical or "." in job_id:
            continue
        try:
            elapsed_seconds = int(elapsed)
            restarts = int(restart_count)
        except ValueError as error:
            raise ValueError("sacct ElapsedRaw and RestartCnt must be integers") from error
        gpus = _allocated_gpus(alloc_tres)
        rows.append(
            {
                "job_id": job_id,
                "logical_job_id": scheduler_to_logical[root],
                "job_name": name,
                "state": state,
                "exit_code": exit_code,
                "classification": classify_state(state, exit_code),
                "elapsed_seconds": elapsed_seconds,
                "allocated_gpus": gpus,
                "gpu_hours": elapsed_seconds * gpus / 3600,
                "restart_count": restarts,
                "submitted_at": _sacct_time(submitted),
                "started_at": _sacct_time(started),
                "ended_at": _sacct_time(ended),
            }
        )
    array_roots = {row["job_id"].split("_", 1)[0] for row in rows if "_" in row["job_id"]}
    return [row for row in rows if row["job_id"] not in array_roots]


def ingest_sacct(submission_path, runtime_root, *, runner=subprocess.run, now=None):
    submission_path = Path(submission_path).expanduser().resolve()
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    job_ids = list(submission["jobs"].values())
    fields = "JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocTRES,RestartCnt,Submit,Start,End"
    result = runner(
        ["sacct", "--noheader", "--parsable2", "--jobs", ",".join(job_ids), "--format", fields],
        check=True,
        capture_output=True,
        text=True,
    )
    collected = now or datetime.now(timezone.utc)
    record = {
        "format": "speck_slurm_sacct",
        "format_version": 1,
        "manifest_sha256": submission["manifest_sha256"],
        "submission": str(submission_path),
        "collected_at_utc": collected.isoformat(),
        "jobs": parse_sacct(result.stdout, submission["jobs"]),
    }
    for job in record["jobs"]:
        job["allocation"] = submission["allocations"][job["logical_job_id"]]
    path = (
        Path(runtime_root).expanduser().resolve()
        / "sacct"
        / submission["manifest_sha256"]
        / f"{collected.isoformat().replace(':', '')}.json"
    )
    _write_exclusive(path, record)
    return record, path


def retry_job(
    manifest_path,
    logical_job_id,
    submission_path,
    sacct_path,
    runtime_root,
    *,
    account=None,
    partition=None,
    runner=subprocess.run,
):
    """Retry a wholly mechanical failure without changing the frozen manifest."""

    manifest, digest, _, _ = load_wave(manifest_path)
    jobs = {job["id"]: job for job in manifest["jobs"]}
    if logical_job_id not in jobs:
        raise ValueError("retry job is not in the manifest")
    job = jobs[logical_job_id]
    if job["allocation"] == "reserve":
        raise ValueError("automatic reserve retry refused")
    submission_path = Path(submission_path).expanduser().resolve()
    sacct_path = Path(sacct_path).expanduser().resolve()
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    observation = json.loads(sacct_path.read_text(encoding="utf-8"))
    if (
        submission.get("manifest_sha256") != digest
        or observation.get("manifest_sha256") != digest
        or observation.get("submission") != str(submission_path)
        or logical_job_id not in submission.get("jobs", {})
    ):
        raise ValueError("retry inputs do not bind the same manifest and submission")
    scheduler_id = submission["jobs"][logical_job_id]
    rows = [
        row
        for row in observation.get("jobs", [])
        if row.get("logical_job_id") == logical_job_id
        and row["job_id"].split("_", 1)[0] == scheduler_id
    ]
    if job["array"]:
        expected_indices = submission.get("retry_task_indices") or job["array"]["indices"]
        try:
            observed_indices = [int(row["job_id"].split("_", 1)[1]) for row in rows]
        except (IndexError, ValueError) as error:
            raise ValueError("array retry rows lack numeric task ids") from error
        if sorted(observed_indices) != sorted(expected_indices):
            raise ValueError("retry requires a complete sacct observation for the submitted array")
    elif len(rows) != 1 or rows[0]["job_id"] != scheduler_id:
        raise ValueError("retry requires one complete sacct observation for the job")
    if not rows or any(
        row.get("classification") not in {"success", "mechanical_retryable"} for row in rows
    ):
        raise ValueError("retry requires success or terminal mechanical-retryable sacct rows")
    failed = [row for row in rows if row["classification"] == "mechanical_retryable"]
    if not failed:
        raise ValueError("retry observation has no mechanical failure")
    task_indices = None
    if job["array"]:
        task_indices = sorted(int(row["job_id"].split("_", 1)[1]) for row in failed)
        if len(task_indices) != len(set(task_indices)) or any(
            index not in job["array"]["indices"] for index in task_indices
        ):
            raise ValueError("array retry task ids do not match the manifest")
    prior_offset = submission.get("attempt_offsets", {}).get(logical_job_id, 0)
    scheduler_attempts = []
    attempt_dir = Path(runtime_root).expanduser().resolve() / "attempts" / digest / logical_job_id
    for attempt_path in attempt_dir.glob(f"{scheduler_id}-*"):
        try:
            scheduler_attempts.append(int(attempt_path.read_text(encoding="utf-8").strip()))
        except ValueError as error:
            raise ValueError("persisted retry state is not an integer") from error
    previous_attempt = max(
        [prior_offset, *scheduler_attempts, *(row.get("restart_count", 0) for row in failed)]
    )
    next_offset = previous_attempt + 1
    if next_offset > job["max_retries"]:
        raise ValueError("same-manifest retry bound is exhausted")
    rendered = render_wave(manifest_path, runtime_root, account=account, partition=partition)
    claim_key = hashlib.sha256(
        f"{digest}:{scheduler_id}:{logical_job_id}:{next_offset}".encode()
    ).hexdigest()
    _write_exclusive(
        Path(runtime_root).expanduser().resolve() / "retry-claims" / digest / f"{claim_key}.json",
        {
            "format": "speck_slurm_retry_claim",
            "format_version": 1,
            "manifest_sha256": digest,
            "logical_job_id": logical_job_id,
            "scheduler_job_id": scheduler_id,
            "attempt_offset": next_offset,
            "submission": str(submission_path),
            "sacct": str(sacct_path),
        },
    )
    command = [
        "sbatch",
        "--parsable",
    ]
    if task_indices is not None:
        retry_array = {**job["array"], "indices": task_indices}
        command.append(f"--array={_array_spec(retry_array)}")
    command.extend(
        [
            f"--export=ALL,SPECK_RETRY_OFFSET={next_offset}",
            rendered["scripts"][logical_job_id],
        ]
    )
    result = runner(command, check=True, capture_output=True, text=True)
    new_scheduler_id = result.stdout.strip().split(";", 1)[0]
    if not new_scheduler_id.isdigit():
        raise ValueError("sbatch did not return a numeric job id")
    record = {
        "format": "speck_slurm_submission",
        "format_version": 1,
        "manifest_sha256": digest,
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "jobs": {logical_job_id: new_scheduler_id},
        "allocations": {logical_job_id: job["allocation"]},
        "attempt_offsets": {logical_job_id: next_offset},
        "retry_task_indices": task_indices,
        "retry_of": {"submission": str(submission_path), "sacct": str(sacct_path)},
        "rendering_id": rendered["rendering_id"],
        "render_options": rendered["render_options"],
        "script_sha256": {logical_job_id: rendered["script_sha256"][logical_job_id]},
        "reserve_spend_automated": False,
        "scientific_promotion_automated": False,
    }
    record_path = (
        Path(runtime_root).expanduser().resolve()
        / "submissions"
        / digest
        / f"{record['submitted_at_utc'].replace(':', '')}.json"
    )
    _write_exclusive(record_path, record)
    return record, record_path


def daily_summary(runtime_root, *, day=None):
    """Summarize the latest sacct observation per scheduler job for one UTC day."""

    target = (
        date.fromisoformat(day)
        if isinstance(day, str)
        else day or datetime.now(timezone.utc).date()
    )
    root = Path(runtime_root).expanduser().resolve()
    latest = {}
    for path in sorted((root / "sacct").glob("*/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        observed = datetime.fromisoformat(record["collected_at_utc"])
        for job in record["jobs"]:
            key = (record["manifest_sha256"], job["job_id"])
            if key not in latest or observed > latest[key][0]:
                latest[key] = (observed, job)
    jobs = [value[1] for value in latest.values()]
    ended_today = [
        job
        for job in jobs
        if job["ended_at"] and datetime.fromisoformat(job["ended_at"]).date() == target
    ]
    classes = {}
    for job in jobs:
        classes[job["classification"]] = classes.get(job["classification"], 0) + 1
    consumed = {
        allocation: sum(job["gpu_hours"] for job in jobs if job.get("allocation") == allocation)
        for allocation in ("mandatory", "reserve")
    }
    total_consumed = sum(consumed.values())
    commitments = _budget_commitments(root)
    return {
        "format": "speck_slurm_daily_summary",
        "format_version": 1,
        "date_utc": target.isoformat(),
        "observed_jobs": len(jobs),
        "ended_today": len(ended_today),
        "classifications": dict(sorted(classes.items())),
        "observed_gpu_hours": consumed,
        "committed_maximum_gpu_hours": commitments,
        "mandatory_ceiling_gpu_hours": MANDATORY_GPU_HOURS,
        "protected_reserve_gpu_hours": RESERVE_GPU_HOURS,
        "mandatory_remaining_gpu_hours": MANDATORY_GPU_HOURS - consumed["mandatory"],
        "reserve_remaining_gpu_hours": RESERVE_GPU_HOURS - consumed["reserve"],
        "unassigned_against_total_gpu_hours": TOTAL_GPU_HOURS - total_consumed,
        "note": "sacct usage is observational; this command does not promote work or authorize reserve",
    }
