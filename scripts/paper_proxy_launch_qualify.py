"""Qualify the frozen Paper 1 proxy boundary and live first-run environment."""

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.helmet_data_acquire import qualify_volume

ROOT = Path(__file__).resolve().parents[1]


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("--output", type=Path, required=True)
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


def command(command, check=True):
    return subprocess.run(command, check=check, capture_output=True, text=True)


def repository_revision(root=ROOT):
    if command(["git", "status", "--porcelain"], check=True).stdout.strip():
        raise ValueError("proxy launch qualification requires a clean repository")
    return command(["git", "rev-parse", "HEAD"], check=True).stdout.strip()


def validate_frozen_inputs(contract, root=ROOT):
    if (
        contract.get("format") != "speck_paper_proxy_launch_contract"
        or contract.get("format_version") != 1
        or contract.get("status") != "frozen_before_first_proxy_result"
    ):
        raise ValueError("proxy launch contract identity is invalid")
    prerequisites = contract.get("prerequisites", ())
    if len(prerequisites) != 6 or len({entry.get("id") for entry in prerequisites}) != 6:
        raise ValueError("proxy launch prerequisites are incomplete or duplicated")
    values = {}
    for entry in prerequisites:
        path = root / entry["path"]
        if not path.is_file() or file_sha256(path) != entry["sha256"]:
            raise ValueError(f"proxy launch prerequisite drifted: {entry['id']}")
        value = load_object(path)
        if value.get("status") != entry["required_status"]:
            raise ValueError(f"proxy launch prerequisite status changed: {entry['id']}")
        values[entry["id"]] = value

    matrix = values["baseline_matrix"]
    analysis = values["analysis_plan"]
    materialization = values["materialization"]
    preflight = values["hardware_preflight"]
    storage = values["storage_qualification"]
    evaluation = values["evaluation_manifest"]
    boundary = contract.get("evaluation_boundary", {})
    execution = contract.get("execution", {})

    if (
        matrix.get("format") != "speck_paper_baseline_matrix"
        or analysis.get("baseline_matrix_sha256") != file_sha256(
            root / "research/paper-1/baseline_matrix.json"
        )
        or materialization.get("contract_sha256") != analysis.get("baseline_matrix_sha256")
        or preflight.get("matrix_sha256") != analysis.get("baseline_matrix_sha256")
        or analysis.get("stopping_rule", {}).get("required_complete_model_runs") != 6
        or analysis.get("stopping_rule", {}).get("interim_efficacy_looks") != 0
        or analysis.get("stopping_rule", {}).get("interim_futility_looks") != 0
        or analysis.get("time_to_quality_target", {}).get("control_only") is not True
    ):
        raise ValueError("proxy baseline design or analysis freeze is invalid")
    if (
        preflight.get("format") != "speck_paper_baseline_preflight"
        or preflight.get("format_version") != 2
        or len(preflight.get("arms", ())) != 2
        or not all(arm.get("passed") is True for arm in preflight["arms"])
    ):
        raise ValueError("proxy hardware preflight is invalid")
    if (
        storage.get("format") != "speck_paper_baseline_storage_qualification"
        or storage.get("capacity", {}).get("proxy_floor_passed") is not True
        or len(storage.get("operational_binding", {}).get("runs", ())) != 6
        or storage.get("operational_binding", {}).get("base_train_sha256")
        != file_sha256(root / "scripts/base_train.py")
    ):
        raise ValueError("proxy storage qualification is invalid")
    if (
        evaluation.get("format") != "speck_architecture_evaluation_manifest"
        or evaluation.get("manifest_id") != "architecture-evaluation-v2"
        or evaluation.get("rules", {}).get("missing_suite")
        != "a required suite that cannot run is a failed release gate, not an omitted metric"
        or set(evaluation.get("release_gate", {}).get("required_external", ()))
        != {"ruler", "nolima", "helmet"}
        or set(
            boundary.get("release_and_capability_claims", {}).get(
                "required_external_suites", ()
            )
        )
        != {"ruler", "nolima", "helmet"}
        or boundary.get("release_and_capability_claims", {}).get("status") != "blocked"
    ):
        raise ValueError("proxy/release evaluation boundary is invalid")

    run_entries = storage["operational_binding"]["runs"]
    runs = {entry["run"]: entry for entry in run_entries}
    controls = execution.get("control_runs", ())
    candidates = execution.get("candidate_runs", ())
    if (
        execution.get("fixed_sample_model_runs") != 6
        or execution.get("control_first") is not True
        or execution.get("interim_quality_decisions") != 0
        or len(controls) != 3
        or len(candidates) != 3
        or set(controls) | set(candidates) != set(runs)
        or any("dense_global_param_match" not in run for run in controls)
        or any("five_cache_kda_gqa" not in run for run in candidates)
    ):
        raise ValueError("proxy execution order does not match the six frozen runs")
    return {"values": values, "runs": runs}


def _gpu_state(expected):
    fields = ["name", "uuid", "memory.total", "utilization.gpu", "temperature.gpu"]
    output = command(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ]
    ).stdout
    rows = list(csv.reader(io.StringIO(output), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != len(fields):
        raise ValueError("expected exactly one visible GPU")
    name, uuid, memory_mib, utilization, temperature = rows[0]
    process_output = command(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    ).stdout.strip()
    processes = [line for line in process_output.splitlines() if line.strip()]
    state = {
        "name": name.strip(),
        "uuid": uuid.strip(),
        "total_memory_bytes": int(memory_mib) * 1024 * 1024,
        "utilization_percent": int(utilization),
        "temperature_c": int(temperature),
        "active_compute_processes": processes,
    }
    if (
        state["name"] != expected["name"]
        or state["uuid"] != expected["uuid"]
        or state["total_memory_bytes"] < expected["minimum_total_memory_bytes"]
        or state["utilization_percent"] > expected["maximum_start_utilization_percent"]
        or state["temperature_c"] > expected["maximum_start_temperature_c"]
        or len(processes) != expected["active_compute_processes"]
    ):
        raise ValueError(f"live GPU does not pass the proxy launch gate: {state}")
    return state


def _memory_state(minimum_available):
    values = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        values[key] = int(raw.strip().split()[0]) * 1024
    state = {
        "total_bytes": values["MemTotal"],
        "available_bytes": values["MemAvailable"],
        "swap_total_bytes": values["SwapTotal"],
        "swap_free_bytes": values["SwapFree"],
    }
    if state["available_bytes"] < minimum_available:
        raise ValueError("host available memory is below the frozen proxy launch floor")
    return state


def _unit_states(units):
    states = {}
    for unit in units:
        result = command(["systemctl", "--user", "is-active", unit], check=False)
        state = result.stdout.strip() or "unknown"
        states[unit] = state
        if state == "active":
            raise ValueError(f"conflicting user unit is active: {unit}")
    return states


def qualify(contract_path, root=ROOT):
    contract_path = Path(contract_path).expanduser().resolve()
    contract = load_object(contract_path)
    frozen = validate_frozen_inputs(contract, root)
    revision = repository_revision(root)
    live = contract["live_gate"]
    storage_expected = live["storage"]
    mount = qualify_volume(
        Path(storage_expected["directory"]), storage_expected["minimum_free_bytes"]
    )
    if (
        mount["uuid"] != storage_expected["mount_uuid"]
        or mount["filesystem"] != storage_expected["filesystem"]
        or not set(storage_expected["required_options"]).issubset(mount["options"])
    ):
        raise ValueError("live checkpoint volume does not match the frozen storage gate")

    run_entries = frozen["runs"]
    existing_targets = sorted(
        run for run, entry in run_entries.items() if Path(entry["checkpoint_directory"]).exists()
    )
    existing_results = sorted(
        run
        for run in run_entries
        if (root / "results/Speck-Paper1/runs" / f"{run}.json").exists()
    )
    if existing_targets or existing_results:
        raise ValueError(
            f"proxy outputs already exist: checkpoints={existing_targets}, results={existing_results}"
        )

    gpu = _gpu_state(live["gpu"])
    memory = _memory_state(live["host"]["minimum_available_memory_bytes"])
    units = _unit_states(live["host"]["required_inactive_user_units"])
    first_run = contract["execution"]["control_runs"][0]
    first = run_entries[first_run]
    return {
        "format": "speck_paper_proxy_launch_qualification",
        "format_version": 1,
        "status": "proxy_training_authorized_release_claims_blocked",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "path": contract_path.relative_to(root).as_posix(),
            "sha256": file_sha256(contract_path),
            "status": contract["status"],
        },
        "repository": {"revision": revision, "clean": True},
        "frozen_inputs": {
            entry["id"]: {"path": entry["path"], "sha256": entry["sha256"]}
            for entry in contract["prerequisites"]
        },
        "live_gate": {
            "gpu": gpu,
            "storage": mount,
            "host_memory": memory,
            "user_units": units,
            "existing_checkpoint_targets": existing_targets,
            "existing_result_records": existing_results,
            "passed": True,
        },
        "next_run": {
            "run": first_run,
            "experiment": first["experiment"],
            "checkpoint_directory": first["checkpoint_directory"],
            "launch": first["launch"],
            "tracking_mode": contract["execution"]["tracking_mode"],
        },
        "decision": {
            "proxy_training_authorized": True,
            "release_claims_authorized": False,
            "long_context_capability_claims_authorized": False,
            "architecture_promotion_authorized": False,
            "paper_scale_pretraining_authorized": False,
            "external_missing_suites_preserved_as_failed_release_gates": True,
        },
        "limitations": contract["evaluation_boundary"]["forbidden_inferences"],
        "runner_revision": revision,
        "runner_sha256": file_sha256(__file__),
    }


def atomic_json(path, value):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv=None):
    args = arguments(argv)
    report = qualify(args.contract)
    atomic_json(args.output, report)
    print(f"Paper 1 proxy launch: {report['status']}")
    print(f"Next run: {report['next_run']['run']}")


if __name__ == "__main__":
    main()
