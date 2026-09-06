"""Plan finalist systems trials and detect protected-tree mutation around child workloads."""

import hashlib
import json
import subprocess
from pathlib import Path

from speck.checkpoint import directory_identity


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


def _within(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
    except ValueError:
        return False
    return True


def require_output_outside_protected(output_path, protected_paths):
    output_path = Path(output_path).expanduser().resolve()
    protected = [Path(path).expanduser().resolve() for path in protected_paths]
    if any(_within(output_path, path) or _within(path, output_path) for path in protected):
        raise ValueError("systems workload output overlaps a protected tree")
    return output_path


def build_trial_plan(protocol_path, qualification_path, output_root):
    protocol_path, protocol = load_object(protocol_path)
    qualification_path, qualification = load_object(qualification_path)
    repository_root = protocol_path.parents[2]
    qualification_reference = protocol.get("inputs", {}).get("run_qualification", {})
    try:
        qualification_relative = qualification_path.relative_to(repository_root).as_posix()
    except ValueError as error:
        raise ValueError(
            "systems workload qualification is outside the frozen repository"
        ) from error
    if (
        protocol.get("format") != "speck_paper_finalist_systems_protocol"
        or protocol.get("format_version") != 2
        or protocol.get("status") != "frozen_before_any_finalist_result_execution_blocked"
        or qualification_reference.get("path") != qualification_relative
        or qualification_reference.get("sha256") != file_sha256(qualification_path)
        or qualification.get("status")
        != "materialization_data_and_storage_qualified_runtime_analysis_and_release_gates_blocked"
    ):
        raise ValueError("systems workload inputs do not match the frozen protocol")
    materialization_reference = protocol["inputs"]["materialization_contract"]
    materialization_path = repository_root / materialization_reference["path"]
    if file_sha256(materialization_path) != materialization_reference["sha256"]:
        raise ValueError("systems workload materialization contract changed")
    _, materialization = load_object(materialization_path)
    checkpoint_root = Path(materialization["identity"]["checkpoint_root"]).resolve()
    output_root = Path(output_root).expanduser().resolve()
    runs = qualification.get("runs")
    if not isinstance(runs, list) or len(runs) != 12:
        raise ValueError("systems workload qualification must contain 12 runs")
    run_map = {
        (
            entry.get("seed"),
            entry.get("data_token_offset"),
            entry.get("run", "").rsplit("-", 1)[-1],
        ): entry
        for entry in runs
    }
    if len(run_map) != 12:
        raise ValueError("systems workload qualification run inventory is ambiguous")
    plans = []
    for block in protocol["paired_blocks"]:
        pair_id = f"pair-{block['pair']}-seed-{block['seed']}-order-{block['data_token_offset']}"
        for position, role in enumerate(block["trial_order"]):
            arm_id = protocol["arms"][role]
            entry = next(
                (
                    value
                    for value in runs
                    if value.get("seed") == block["seed"]
                    and value.get("data_token_offset") == block["data_token_offset"]
                    and value.get("run", "").endswith(f"-{arm_id}")
                ),
                None,
            )
            if entry is None:
                raise ValueError("systems workload run is absent from qualification")
            expected_experiment = (
                Path("experiments") / "Speck-Paper1-Finalist-131M" / "runs" / pair_id / arm_id
            )
            checkpoint = Path(entry["checkpoint_directory"]).resolve()
            if (
                entry.get("experiment") != expected_experiment.as_posix()
                or checkpoint.parent != checkpoint_root
                or checkpoint.name != entry.get("run")
            ):
                raise ValueError("systems workload experiment or checkpoint path drifted")
            experiment = (repository_root / expected_experiment).resolve()
            if not experiment.is_dir():
                raise ValueError("systems workload experiment directory is absent")
            output = output_root / f"block-{block['block']}-{position}-{role}.json"
            require_output_outside_protected(output, (experiment, checkpoint))
            plans.append(
                {
                    "block": block["block"],
                    "pair": block["pair"],
                    "position": position,
                    "role": role,
                    "arm_id": arm_id,
                    "run": entry["run"],
                    "experiment": str(experiment),
                    "checkpoint_directory": str(checkpoint),
                    "checkpoint_step": protocol["trial_workload"]["checkpoint_step"],
                    "benchmark_data_start": block["benchmark_data_start"],
                    "benchmark_data_end": block["benchmark_data_end"],
                    "warmup_optimizer_steps": protocol["trial_workload"]["warmup_optimizer_steps"],
                    "measured_optimizer_steps": protocol["trial_workload"][
                        "measured_optimizer_steps"
                    ],
                    "measured_tokens": protocol["trial_workload"]["measured_tokens_per_trial"],
                    "output": str(output),
                    "checkpoint_write_authorized": False,
                    "model_or_optimizer_output_persisted": False,
                    "execution_authorized": False,
                }
            )
    if (
        len(plans) != 12
        or len({plan["run"] for plan in plans}) != 12
        or len({plan["checkpoint_directory"] for plan in plans}) != 12
        or len({plan["output"] for plan in plans}) != 12
    ):
        raise ValueError("systems workload trial plan is not one-to-one")
    return {
        "format": "speck_paper_finalist_systems_workload_plan",
        "format_version": 1,
        "status": "planned_execution_blocked",
        "protocol": {"path": str(protocol_path), "sha256": file_sha256(protocol_path)},
        "qualification": {
            "path": str(qualification_path),
            "sha256": file_sha256(qualification_path),
        },
        "trials": plans,
        "execution_authorized": False,
    }


def protected_identity(path):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        identity = directory_identity(path)
        return {
            "path": str(path),
            "kind": "directory",
            "sha256": identity["sha256"],
            "files": len(identity["files"]),
            "bytes": sum(entry["bytes"] for entry in identity["files"]),
        }
    if path.is_file():
        return {
            "path": str(path),
            "kind": "file",
            "sha256": file_sha256(path),
            "files": 1,
            "bytes": path.stat().st_size,
        }
    raise ValueError(f"protected systems workload path is absent: {path}")


def guarded_run(command, protected_paths, output_path, *, runner=None):
    """Run a child and reject any before/after protected-tree identity change."""

    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(value, str) for value in command)
    ):
        raise ValueError("systems workload command is invalid")
    protected_paths = [Path(path).expanduser().resolve() for path in protected_paths]
    if len(set(protected_paths)) != len(protected_paths):
        raise ValueError("systems workload protected paths are duplicated")
    output_path = require_output_outside_protected(output_path, protected_paths)
    before = [protected_identity(path) for path in protected_paths]
    runner = runner or (
        lambda values: subprocess.run(values, check=False, capture_output=True, text=True)
    )
    child_error = None
    result = None
    try:
        result = runner(command)
    except Exception as error:
        child_error = error
    try:
        after = [protected_identity(path) for path in protected_paths]
    except Exception as identity_error:
        raise RuntimeError("systems workload mutated a protected tree") from (
            child_error or identity_error
        )
    if before != after:
        raise RuntimeError("systems workload mutated a protected tree") from child_error
    if child_error is not None:
        raise child_error
    if not isinstance(result.returncode, int) or result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)
    return {
        "command": command,
        "returncode": result.returncode,
        "output": str(output_path),
        "protected_before": before,
        "protected_after": after,
        "mutation_detected": False,
        "kernel_read_only_enforced": False,
    }
