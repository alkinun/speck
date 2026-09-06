"""Validate all finalist systems blocks before invoking the frozen analyzer."""

import hashlib
import json
from pathlib import Path

from speck.paper_finalist_systems_analysis import analyze_systems
from speck.paper_finalist_systems_assembly import SOFTWARE_IDENTITY_FIELDS


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


def _resolve(root, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _validate_complete_block(block_path, report, expected, protocol, artifact_root):
    if (
        report.get("trial_order") != expected["trial_order"]
        or report.get("pair") != expected["pair"]
        or report.get("replacement_or_retry_authorized") is not False
        or set(report.get("trials", {})) != {"control", "candidate"}
    ):
        raise ValueError("systems accepted block identity is invalid")
    references = report.get("trial_references")
    if not isinstance(references, list) or len(references) != 2:
        raise ValueError("systems accepted block trial references are incomplete")
    reference_by_role = {reference.get("role"): reference for reference in references}
    if set(reference_by_role) != {"control", "candidate"}:
        raise ValueError("systems accepted block trial reference roles are invalid")
    trials = report["trials"]
    for position, role in enumerate(expected["trial_order"]):
        trial = trials[role]
        reference = reference_by_role[role]
        path = _resolve(artifact_root, reference.get("path", ""))
        if (
            not path.is_file()
            or file_sha256(path) != reference.get("sha256")
            or load_object(path)[1] != trial
            or trial.get("format") != "speck_paper_finalist_systems_trial_result"
            or trial.get("format_version") != 1
            or trial.get("status") != "complete_qualified"
            or trial.get("block") != expected["block"]
            or trial.get("pair") != expected["pair"]
            or trial.get("position") != position
            or trial.get("role") != role
            or trial.get("arm_id") != protocol["arms"][role]
            or set(trial.get("software_identities", {})) != SOFTWARE_IDENTITY_FIELDS
            or not isinstance(trial.get("paired_batches", {}).get("sha256"), str)
            or len(trial.get("paired_batches", {}).get("sha256", "")) != 64
        ):
            raise ValueError("systems accepted block trial bytes or identity are invalid")
    if report.get("paired_batch_sha256") != trials["control"].get("paired_batches", {}).get(
        "sha256"
    ) or report.get("paired_batch_sha256") != trials["candidate"].get("paired_batches", {}).get(
        "sha256"
    ):
        raise ValueError("systems accepted block paired data identity is invalid")
    return trials


def validate_systems_evidence(protocol_path, block_paths, artifact_root=None):
    protocol_path, protocol = load_object(protocol_path)
    artifact_root = Path(artifact_root).resolve() if artifact_root else protocol_path.parents[2]
    protocol_sha = file_sha256(protocol_path)
    expected = {block["block"]: block for block in protocol["paired_blocks"]}
    observed = {}
    references = []
    for block_path in block_paths:
        path, report = load_object(block_path)
        block = report.get("block")
        if block not in expected or block in observed:
            raise ValueError("systems acceptance found an unexpected or duplicate block")
        if (
            report.get("format") != "speck_paper_finalist_systems_block_result"
            or report.get("format_version") != 1
            or report.get("protocol_sha256") != protocol_sha
        ):
            raise ValueError("systems acceptance block does not match the frozen protocol")
        observed[block] = (path, report)
        references.append({"path": str(path), "sha256": file_sha256(path), "block": block})
    missing = sorted(set(expected) - set(observed))
    failed = sorted(
        block
        for block, (_, report) in observed.items()
        if report.get("status") == "failed_retained"
    )
    invalid_status = [
        block
        for block, (_, report) in observed.items()
        if report.get("status") not in {"complete_qualified", "failed_retained"}
    ]
    if invalid_status:
        raise ValueError("systems acceptance block has an invalid status")
    complete_trials = {}
    for block, (path, report) in observed.items():
        if report.get("status") == "failed_retained":
            if (
                not report.get("failure")
                or report.get("replacement_or_retry_authorized") is not False
            ):
                raise ValueError("systems acceptance retained failure is incomplete")
        else:
            complete_trials[block] = _validate_complete_block(
                path,
                report,
                expected[block],
                protocol,
                artifact_root,
            )
    if missing or failed:
        return {
            "status": "incomplete_or_failed_no_systems_claim",
            "block_references": sorted(references, key=lambda value: value["block"]),
            "missing_blocks": missing,
            "failed_blocks": failed,
            "valid_complete_blocks": sorted(set(observed) - set(failed)),
            "analysis_executed": False,
            "joint_training_systems_efficiency_pass": False,
            "execution_or_replacement_authorized": False,
        }
    trials = [
        trial for block in sorted(complete_trials) for trial in complete_trials[block].values()
    ]
    if len(trials) != 12:
        raise ValueError("systems acceptance does not contain exactly 12 trials")
    common_fields = SOFTWARE_IDENTITY_FIELDS - {"model_config_sha256"}
    common = {
        field: {trial["software_identities"][field] for trial in trials} for field in common_fields
    }
    if any(len(values) != 1 for values in common.values()):
        raise ValueError("systems acceptance found global software identity drift")
    model_by_role = {
        role: {
            trial["software_identities"]["model_config_sha256"]
            for trial in trials
            if trial["role"] == role
        }
        for role in ("control", "candidate")
    }
    if (
        any(len(values) != 1 for values in model_by_role.values())
        or model_by_role["control"] == model_by_role["candidate"]
    ):
        raise ValueError("systems acceptance found invalid arm-specific model identities")
    analysis = analyze_systems(protocol_path, [observed[index][0] for index in sorted(observed)])
    return {
        "status": "complete_valid",
        "block_references": sorted(references, key=lambda value: value["block"]),
        "missing_blocks": [],
        "failed_blocks": [],
        "valid_complete_blocks": list(range(6)),
        "trials": 12,
        "common_software_identities": {
            field: next(iter(values)) for field, values in common.items()
        },
        "model_config_sha256_by_role": {
            role: next(iter(values)) for role, values in model_by_role.items()
        },
        "analysis_executed": True,
        "analysis": analysis,
        "joint_training_systems_efficiency_pass": analysis[
            "joint_training_systems_efficiency_pass"
        ],
        "execution_or_replacement_authorized": False,
    }
