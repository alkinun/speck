"""Durably orchestrate and qualify the future 20B production data rehearsal."""

import hashlib
import json
import os
import resource
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

FORMAT = "speck_data_rehearsal"
FORMAT_VERSION = 1
MANIFEST_FORMAT = "speck_data_rehearsal_result"
STAGE_FORMAT = "speck_data_rehearsal_stage_result"
STAGE_IDS = (
    "source_identity",
    "acquisition",
    "global_dedup",
    "packing",
    "resume_cleanup",
    "firewall_disjointness",
)
REQUIRED_GATES = (
    "source_identity",
    "global_exact_deduplication",
    "global_near_deduplication",
    "packing_integrity",
    "acquisition_cleanup",
    "interruption_resume",
    "firewall_training_disjointness",
)
REQUIRED_METRICS = (
    "download_bytes",
    "download_bytes_per_second",
    "filtered_bytes",
    "records_seen",
    "records_retained",
    "sqlite_index_bytes",
    "peak_rss_bytes",
    "packing_tokens_per_second",
    "packed_tokens",
    "packed_bytes",
    "unique_tokens",
)


def _sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exact_keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{name} must contain exactly: {', '.join(sorted(expected))}")


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _path(value, config_dir, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(value).expanduser()
    return (config_dir / path).resolve() if not path.is_absolute() else path.resolve()


def _identity(value, config_dir, name, *, nullable=False):
    if nullable and value is None:
        return None
    _exact_keys(value, {"path", "sha256"}, name)
    digest = value["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{name} sha256 must be lowercase SHA-256")
    return {"path": str(_path(value["path"], config_dir, name)), "sha256": digest}


def validate_rehearsal_config(config, *, config_dir=None):
    """Validate fixed stage order, authority mode, and required measurements."""

    config_dir = Path(config_dir or ".").resolve()
    _exact_keys(
        config,
        {
            "format",
            "format_version",
            "status",
            "mode",
            "target_tokens",
            "rights_record",
            "deny_ledger",
            "stages",
            "required_gates",
            "required_metrics",
            "output_directory",
        },
        "data rehearsal",
    )
    if config["format"] != FORMAT or config["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported data rehearsal format")
    if config["status"] != "orchestration_authorized_not_training_authority":
        raise ValueError("data rehearsal must remain non-authoritative")
    if config["mode"] not in {"fixture_only", "production_20B"}:
        raise ValueError("rehearsal mode must be fixture_only or production_20B")
    target_tokens = _integer(config["target_tokens"], "target_tokens", 1)
    rights = _identity(config["rights_record"], config_dir, "rights record", nullable=True)
    if config["mode"] == "fixture_only" and rights is not None:
        raise ValueError("fixture rehearsal cannot name a human rights record")
    if config["mode"] == "production_20B" and (target_tokens != 20_000_000_000 or rights is None):
        raise ValueError("production rehearsal requires exactly 20B tokens and a rights record")
    deny = _identity(config["deny_ledger"], config_dir, "deny ledger")
    stages = config["stages"]
    if not isinstance(stages, list) or tuple(stage.get("id") for stage in stages) != STAGE_IDS:
        raise ValueError("rehearsal stages must follow the frozen six-stage order")
    normalized_stages = []
    result_paths = []
    for stage in stages:
        stage_id = stage["id"]
        _exact_keys(
            stage,
            {"id", "command", "cwd", "environment", "inputs", "result"},
            f"stage {stage_id}",
        )
        command = stage["command"]
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(value, str) or not value for value in command)
        ):
            raise ValueError(f"stage {stage_id} command must be non-empty strings")
        environment = stage["environment"]
        if not isinstance(environment, dict) or any(
            not isinstance(key, str) or not key or not isinstance(value, str)
            for key, value in environment.items()
        ):
            raise ValueError(f"stage {stage_id} environment must map strings")
        inputs = stage["inputs"]
        if not isinstance(inputs, list):
            raise ValueError(f"stage {stage_id} inputs must be a list")
        _exact_keys(stage["result"], {"path"}, f"stage {stage_id} result")
        normalized_stages.append(
            {
                **stage,
                "cwd": str(_path(stage["cwd"], config_dir, f"stage {stage_id} cwd")),
                "inputs": [
                    _identity(value, config_dir, f"stage {stage_id} input") for value in inputs
                ],
                "result": {
                    "path": str(
                        _path(
                            stage["result"]["path"],
                            config_dir,
                            f"stage {stage_id} result",
                        )
                    )
                },
            }
        )
        result_paths.append(normalized_stages[-1]["result"]["path"])
    if len(result_paths) != len(set(result_paths)):
        raise ValueError("rehearsal stage result paths must be unique")
    protected = {
        deny["path"],
        *(identity["path"] for stage in normalized_stages for identity in stage["inputs"]),
    }
    if rights is not None:
        protected.add(rights["path"])
    if any(path in protected for path in result_paths):
        raise ValueError("stage results cannot overwrite rehearsal authority or input files")
    if config["required_gates"] != list(REQUIRED_GATES):
        raise ValueError("required rehearsal gates differ from the frozen contract")
    if config["required_metrics"] != list(REQUIRED_METRICS):
        raise ValueError("required rehearsal metrics differ from the frozen contract")
    normalized = {
        **config,
        "target_tokens": target_tokens,
        "rights_record": rights,
        "deny_ledger": deny,
        "stages": normalized_stages,
        "output_directory": str(_path(config["output_directory"], config_dir, "output directory")),
    }
    normalized["plan_fingerprint"] = _fingerprint(normalized)
    return normalized


def load_rehearsal_config(path):
    path = Path(path).resolve()
    return validate_rehearsal_config(json.loads(path.read_text()), config_dir=path.parent)


def _verify_identity(identity, name):
    path = Path(identity["path"])
    if not path.is_file() or _sha256(path) != identity["sha256"]:
        raise ValueError(f"{name} identity mismatch")
    return path


def _tree_bytes(path):
    path = Path(path)
    return sum(value.stat().st_size for value in path.rglob("*") if value.is_file())


def _verify_stage(stage, completed, log_root):
    result_path = _verify_identity(completed["result"], f"completed stage {stage['id']} result")
    for key in ("stdout", "stderr"):
        identity = {
            **completed[key],
            "path": str(Path(log_root) / completed[key]["path"]),
        }
        _verify_identity(identity, f"completed stage {stage['id']} {key}")
    result = json.loads(result_path.read_text())
    if (
        result.get("format") != STAGE_FORMAT
        or result.get("format_version") != FORMAT_VERSION
        or result.get("status") != "complete"
        or result.get("stage_id") != stage["id"]
        or not isinstance(result.get("gates"), dict)
        or not isinstance(result.get("metrics"), dict)
    ):
        raise ValueError(f"stage {stage['id']} result schema is invalid")
    return result


def verify_rehearsal_manifest(manifest_path):
    """Rehash every completed stage and reconstruct aggregate gates and metrics."""

    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_file():
        raise ValueError("rehearsal manifest is missing")
    manifest = json.loads(manifest_path.read_text())
    expected_status = {
        "fixture_only": "fixture_rehearsal_complete_not_production_authority",
        "production_20B": "production_20B_rehearsal_complete_pending_operations_record",
    }
    if (
        manifest.get("format") != MANIFEST_FORMAT
        or manifest.get("format_version") != FORMAT_VERSION
        or manifest.get("status") != expected_status.get(manifest.get("mode"))
        or [stage.get("id") for stage in manifest.get("stages", [])] != list(STAGE_IDS)
    ):
        raise ValueError("rehearsal manifest schema or stage order is invalid")
    gates = {}
    metrics = {}
    for completed in manifest["stages"]:
        result_path = _verify_identity(
            completed["result"], f"manifest stage {completed['id']} result"
        )
        for key in ("stdout", "stderr"):
            identity = {
                **completed[key],
                "path": str(manifest_path.parent / completed[key]["path"]),
            }
            _verify_identity(identity, f"manifest stage {completed['id']} {key}")
        result = json.loads(result_path.read_text())
        if (
            result.get("format") != STAGE_FORMAT
            or result.get("format_version") != FORMAT_VERSION
            or result.get("status") != "complete"
            or result.get("stage_id") != completed["id"]
            or not isinstance(result.get("gates"), dict)
            or not isinstance(result.get("metrics"), dict)
        ):
            raise ValueError(f"manifest stage {completed['id']} result is invalid")
        gates.update(result["gates"])
        metrics.update(result["metrics"])
    if gates != manifest.get("gates") or metrics != manifest.get("metrics"):
        raise ValueError("rehearsal manifest aggregate gates or metrics changed")
    if any(gates.get(gate) != "pass" for gate in REQUIRED_GATES) or any(
        metric not in metrics for metric in REQUIRED_METRICS
    ):
        raise ValueError("rehearsal manifest misses required gates or metrics")
    deny_path = _verify_identity(manifest["deny_ledger"], "manifest deny ledger")
    deny = json.loads(deny_path.read_text())
    if deny.get("status") != "human_reviewed_deny_entries":
        raise ValueError("manifest deny ledger is not human reviewed")
    if manifest["mode"] == "production_20B":
        if manifest.get("target_tokens") != 20_000_000_000 or manifest.get("rights_record") is None:
            raise ValueError("production rehearsal manifest is not exactly 20B and rights-bound")
        rights_path = _verify_identity(manifest["rights_record"], "manifest rights record")
        rights = json.loads(rights_path.read_text())
        if (
            rights.get("status") != "all_sources_human_approved"
            or rights.get("automated_approval_made") is not False
        ):
            raise ValueError("production rehearsal manifest rights are invalid")
    return manifest


def run_rehearsal(config, *, restart=False, crash_after_stage=None):
    """Run or resume fixed stages and capture immutable logs and host telemetry."""

    if "plan_fingerprint" not in config:
        config = validate_rehearsal_config(config)
    else:
        payload = {key: value for key, value in config.items() if key != "plan_fingerprint"}
        if config["plan_fingerprint"] != _fingerprint(payload):
            raise ValueError("normalized rehearsal fingerprint mismatch")
    if config["rights_record"] is not None:
        path = _verify_identity(config["rights_record"], "rights record")
        rights = json.loads(path.read_text())
        if (
            rights.get("format") != "speck_human_source_rights_acceptance"
            or rights.get("status") != "all_sources_human_approved"
            or rights.get("automated_approval_made") is not False
        ):
            raise ValueError("production rehearsal human rights record is invalid")
    deny_path = _verify_identity(config["deny_ledger"], "deny ledger")
    deny = json.loads(deny_path.read_text())
    if (
        deny.get("format") != "speck_removal_deny_ledger"
        or deny.get("status") != "human_reviewed_deny_entries"
    ):
        raise ValueError("rehearsal deny ledger is invalid")
    output = Path(config["output_directory"])
    staging = output.with_name(output.name + ".building")
    state_path = staging / "state.json"
    if output.exists():
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("plan_fingerprint") != config["plan_fingerprint"]:
            raise ValueError("published rehearsal has a different plan")
        return verify_rehearsal_manifest(manifest_path)
    if staging.exists() and restart:
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {
            "format": "speck_data_rehearsal_state",
            "format_version": FORMAT_VERSION,
            "contract": config["plan_fingerprint"],
            "completed": [],
        }
    )
    if state.get("contract") != config["plan_fingerprint"]:
        raise ValueError("staged rehearsal plan changed; use restart")
    for index, completed in enumerate(state["completed"]):
        if completed["id"] != config["stages"][index]["id"]:
            raise ValueError("completed rehearsal stages are not an ordered prefix")
        _verify_stage(config["stages"][index], completed, staging)
    aggregate_gates = {}
    aggregate_metrics = {}
    for completed in state["completed"]:
        result = _verify_stage(
            next(stage for stage in config["stages"] if stage["id"] == completed["id"]),
            completed,
            staging,
        )
        aggregate_gates.update(result["gates"])
        aggregate_metrics.update(result["metrics"])
    for stage in config["stages"][len(state["completed"]) :]:
        for identity in stage["inputs"]:
            _verify_identity(identity, f"stage {stage['id']} input")
        result_path = Path(stage["result"]["path"])
        result_path.unlink(missing_ok=True)
        stdout_path = staging / f"{stage['id']}.stdout"
        stderr_path = staging / f"{stage['id']}.stderr"
        usage_before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        free_before = shutil.disk_usage(staging).free
        tree_before = _tree_bytes(staging)
        started = time.perf_counter()
        environment = os.environ.copy()
        environment.update(stage["environment"])
        with (
            stdout_path.open("wb") as stdout,
            stderr_path.open("wb") as stderr,
        ):
            completed_process = subprocess.run(
                stage["command"],
                cwd=stage["cwd"],
                env=environment,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
            stdout.flush()
            stderr.flush()
            os.fsync(stdout.fileno())
            os.fsync(stderr.fileno())
        elapsed = time.perf_counter() - started
        if completed_process.returncode:
            raise RuntimeError(
                f"rehearsal stage {stage['id']} failed with {completed_process.returncode}"
            )
        result_path = Path(stage["result"]["path"])
        if not result_path.is_file():
            raise ValueError(f"stage {stage['id']} did not produce its result")
        result = json.loads(result_path.read_text())
        completed = {
            "id": stage["id"],
            "command": stage["command"],
            "inputs": stage["inputs"],
            "result": {"path": str(result_path), "sha256": _sha256(result_path)},
            "stdout": {"path": stdout_path.name, "sha256": _sha256(stdout_path)},
            "stderr": {"path": stderr_path.name, "sha256": _sha256(stderr_path)},
            "telemetry": {
                "elapsed_seconds": elapsed,
                "cumulative_child_peak_rss_kib_before": usage_before,
                "cumulative_child_peak_rss_kib_after": resource.getrusage(
                    resource.RUSAGE_CHILDREN
                ).ru_maxrss,
                "filesystem_free_bytes_before": free_before,
                "filesystem_free_bytes_after": shutil.disk_usage(staging).free,
                "staging_tree_bytes_before": tree_before,
                "staging_tree_bytes_after": _tree_bytes(staging),
            },
        }
        result = _verify_stage(stage, completed, staging)
        aggregate_gates.update(result["gates"])
        aggregate_metrics.update(result["metrics"])
        state["completed"].append(completed)
        _write_json(state_path, state)
        if crash_after_stage == stage["id"]:
            raise RuntimeError("injected rehearsal orchestrator crash")
    missing_gates = [gate for gate in REQUIRED_GATES if aggregate_gates.get(gate) != "pass"]
    if missing_gates:
        raise RuntimeError(f"rehearsal required gates did not pass: {missing_gates}")
    missing_metrics = [metric for metric in REQUIRED_METRICS if metric not in aggregate_metrics]
    if missing_metrics or any(
        isinstance(aggregate_metrics.get(metric), bool)
        or not isinstance(aggregate_metrics.get(metric), (int, float))
        or aggregate_metrics[metric] < 0
        for metric in REQUIRED_METRICS
    ):
        raise RuntimeError(f"rehearsal required metrics are missing or invalid: {missing_metrics}")
    if aggregate_metrics["packed_tokens"] < config["target_tokens"]:
        raise RuntimeError("rehearsal packed token count does not reach its target")
    manifest = {
        "format": MANIFEST_FORMAT,
        "format_version": FORMAT_VERSION,
        "status": (
            "fixture_rehearsal_complete_not_production_authority"
            if config["mode"] == "fixture_only"
            else "production_20B_rehearsal_complete_pending_operations_record"
        ),
        "plan_fingerprint": config["plan_fingerprint"],
        "mode": config["mode"],
        "target_tokens": config["target_tokens"],
        "rights_record": config["rights_record"],
        "deny_ledger": config["deny_ledger"],
        "stages": state["completed"],
        "gates": aggregate_gates,
        "metrics": aggregate_metrics,
        "training_authority": "blocked",
    }
    _write_json(staging / "manifest.json", manifest)
    state_path.unlink()
    staging.replace(output)
    return manifest


def issue_operations_qualification(manifest_path, output_path):
    """Emit production operations authority only from an exact completed 20B manifest."""

    manifest_path = Path(manifest_path).resolve()
    manifest = verify_rehearsal_manifest(manifest_path)
    if (
        manifest.get("format") != MANIFEST_FORMAT
        or manifest.get("status") != "production_20B_rehearsal_complete_pending_operations_record"
        or manifest.get("mode") != "production_20B"
        or manifest.get("target_tokens") != 20_000_000_000
        or any(manifest.get("gates", {}).get(gate) != "pass" for gate in REQUIRED_GATES)
        or any(metric not in manifest.get("metrics", {}) for metric in REQUIRED_METRICS)
        or manifest.get("metrics", {}).get("packed_tokens", 0) < 20_000_000_000
    ):
        raise ValueError("only a complete production 20B rehearsal can issue operations authority")
    record = {
        "format": "speck_production_data_operations_qualification",
        "format_version": FORMAT_VERSION,
        "status": "production_data_operations_qualified",
        "qualified_at": datetime.now(timezone.utc).isoformat(),
        "rehearsal_manifest": str(manifest_path),
        "rehearsal_manifest_sha256": _sha256(manifest_path),
        "deny_ledger_sha256": manifest["deny_ledger"]["sha256"],
        "gates": {
            "global_exact_deduplication": "pass",
            "global_near_deduplication": "pass",
            "acquisition_cleanup": "pass",
            "interruption_resume": "pass",
            "firewall_training_disjointness": "pass",
        },
        "metrics": manifest["metrics"],
    }
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(f"operations qualification already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, record)
    return record
