import json
from copy import deepcopy
from pathlib import Path

import pytest

from speck.paper_finalist_systems_acceptance import validate_systems_evidence
from speck.paper_finalist_systems_telemetry import atomic_json, file_sha256
from tests.test_paper_finalist_systems_analysis import trial

root = Path(__file__).parents[1]
protocol_path = root / "research" / "paper-1" / "finalist_systems_v2.json"
protocol = json.loads(protocol_path.read_text(encoding="utf-8"))


def software(role):
    return {
        "git_commit": "a" * 40,
        "python": "3.14",
        "pytorch": "2.9",
        "cuda_runtime": "13.0",
        "cuda_driver": "610.43.03",
        "fla": "0.4",
        "triton": "3.5",
        "model_config_sha256": ("c" if role == "control" else "d") * 64,
        "benchmark_engine_sha256": "e" * 64,
    }


def write_complete_blocks(tmp_path):
    paths = []
    protocol_sha = file_sha256(protocol_path)
    for expected in protocol["paired_blocks"]:
        trials = {}
        references = []
        fingerprint = f"{expected['block']:064x}"
        for position, role in enumerate(expected["trial_order"]):
            value = trial(
                role,
                10 if role == "control" else 8,
                1000 if role == "control" else 800,
                800 if role == "control" else 600,
            )
            value.update(
                {
                    "format": "speck_paper_finalist_systems_trial_result",
                    "format_version": 1,
                    "block": expected["block"],
                    "pair": expected["pair"],
                    "position": position,
                    "role": role,
                    "paired_batches": {"sha256": fingerprint},
                    "software_identities": software(role),
                }
            )
            trial_path = tmp_path / f"block-{expected['block']}-{role}.json"
            atomic_json(trial_path, value)
            trials[role] = value
            references.append(
                {"path": str(trial_path), "sha256": file_sha256(trial_path), "role": role}
            )
        block = {
            "format": "speck_paper_finalist_systems_block_result",
            "format_version": 1,
            "status": "complete_qualified",
            "protocol_sha256": protocol_sha,
            "block": expected["block"],
            "pair": expected["pair"],
            "trial_order": expected["trial_order"],
            "trial_references": references,
            "paired_batch_sha256": fingerprint,
            "trials": trials,
            "replacement_or_retry_authorized": False,
        }
        block_path = tmp_path / f"block-{expected['block']}.json"
        atomic_json(block_path, block)
        paths.append(block_path)
    return paths


def rewrite_trial(block_path, role, transform):
    block = json.loads(block_path.read_text(encoding="utf-8"))
    reference = next(value for value in block["trial_references"] if value["role"] == role)
    trial_path = Path(reference["path"])
    value = json.loads(trial_path.read_text(encoding="utf-8"))
    transform(value)
    atomic_json(trial_path, value)
    reference["sha256"] = file_sha256(trial_path)
    block["trials"][role] = value
    atomic_json(block_path, block)


def test_global_systems_acceptance_runs_analysis_after_all_six_blocks(tmp_path):
    report = validate_systems_evidence(protocol_path, write_complete_blocks(tmp_path), tmp_path)
    assert report["status"] == "complete_valid"
    assert report["trials"] == 12
    assert report["analysis_executed"] is True
    assert report["joint_training_systems_efficiency_pass"] is True
    assert len(report["common_software_identities"]) == 8
    assert (
        report["model_config_sha256_by_role"]["control"]
        != report["model_config_sha256_by_role"]["candidate"]
    )


def test_global_systems_acceptance_rejects_trial_file_drift(tmp_path):
    paths = write_complete_blocks(tmp_path)
    block = json.loads(paths[0].read_text(encoding="utf-8"))
    trial_path = Path(block["trial_references"][0]["path"])
    value = json.loads(trial_path.read_text(encoding="utf-8"))
    value["measured_wall_seconds"] += 1
    atomic_json(trial_path, value)
    with pytest.raises(ValueError, match="bytes or identity"):
        validate_systems_evidence(protocol_path, paths, tmp_path)


def test_global_systems_acceptance_rejects_software_drift(tmp_path):
    paths = write_complete_blocks(tmp_path)
    rewrite_trial(
        paths[4],
        "candidate",
        lambda value: value["software_identities"].update(pytorch="different"),
    )
    with pytest.raises(ValueError, match="global software identity drift"):
        validate_systems_evidence(protocol_path, paths, tmp_path)


def test_global_systems_acceptance_rejects_shared_arm_model_identity(tmp_path):
    paths = write_complete_blocks(tmp_path)
    for path in paths:
        rewrite_trial(
            path,
            "candidate",
            lambda value: value["software_identities"].update(model_config_sha256="c" * 64),
        )
    with pytest.raises(ValueError, match="arm-specific model"):
        validate_systems_evidence(protocol_path, paths, tmp_path)


def test_global_systems_acceptance_retains_failed_block_without_analysis(tmp_path):
    paths = write_complete_blocks(tmp_path)
    failed = deepcopy(json.loads(paths[2].read_text(encoding="utf-8")))
    failed["status"] = "failed_retained"
    failed["failure"] = "runtime_attestation_failure"
    failed.pop("trials")
    atomic_json(paths[2], failed)
    report = validate_systems_evidence(protocol_path, paths, tmp_path)
    assert report["status"] == "incomplete_or_failed_no_systems_claim"
    assert report["failed_blocks"] == [2]
    assert report["analysis_executed"] is False
    assert report["joint_training_systems_efficiency_pass"] is False


def test_failed_block_cannot_mask_drift_in_another_complete_block(tmp_path):
    paths = write_complete_blocks(tmp_path)
    failed = json.loads(paths[2].read_text(encoding="utf-8"))
    failed["status"] = "failed_retained"
    failed["failure"] = "runtime_attestation_failure"
    failed.pop("trials")
    atomic_json(paths[2], failed)
    first = json.loads(paths[0].read_text(encoding="utf-8"))
    Path(first["trial_references"][0]["path"]).write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bytes or identity"):
        validate_systems_evidence(protocol_path, paths, tmp_path)


def test_global_systems_acceptance_retains_missing_block_without_analysis(tmp_path):
    paths = write_complete_blocks(tmp_path)
    report = validate_systems_evidence(protocol_path, paths[:-1], tmp_path)
    assert report["missing_blocks"] == [5]
    assert report["analysis_executed"] is False
    assert report["execution_or_replacement_authorized"] is False
