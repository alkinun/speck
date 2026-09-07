"""Issue and verify fail-closed flagship data launch receipts."""

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from speck.dataloader import manifest_fingerprint

FORMAT = "speck_flagship_data_launch_request"
FORMAT_VERSION = 1
RECEIPT_FORMAT = "speck_flagship_data_launch_receipt"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


def _sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _exact_keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{name} must contain exactly: {', '.join(sorted(expected))}")


def _path(value, config_dir, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(value).expanduser()
    return (config_dir / path).resolve() if not path.is_absolute() else path.resolve()


def _identity(value, config_dir, name):
    _exact_keys(value, {"path", "sha256"}, name)
    if not isinstance(value["sha256"], str) or not _SHA256.fullmatch(value["sha256"]):
        raise ValueError(f"{name} sha256 must be lowercase SHA-256")
    return {"path": str(_path(value["path"], config_dir, name)), "sha256": value["sha256"]}


def validate_launch_request(request, *, config_dir=None):
    """Validate one exact launch request without treating it as authority."""

    config_dir = Path(config_dir or ".").resolve()
    _exact_keys(
        request,
        {
            "format",
            "format_version",
            "status",
            "repository",
            "git_commit",
            "experiment_directory",
            "experiment_files",
            "rights_record",
            "production_operations_record",
            "deny_ledger",
            "firewall_manifest",
            "tokenizer_decision",
            "packed_manifest",
            "receipt_path",
        },
        "data launch request",
    )
    if (
        request["format"] != FORMAT
        or request["format_version"] != FORMAT_VERSION
        or request["status"] != "preflight_requested_not_authority"
    ):
        raise ValueError("unsupported data launch request")
    repository = _path(request["repository"], config_dir, "repository")
    if not _COMMIT.fullmatch(request["git_commit"]):
        raise ValueError("git_commit must be a full lowercase commit")
    experiment = _path(request["experiment_directory"], config_dir, "experiment directory")
    files = request["experiment_files"]
    if not isinstance(files, dict) or set(files) != {"data", "tokenizer", "model", "train"}:
        raise ValueError("experiment_files must bind data/tokenizer/model/train")
    normalized_files = {
        key: _identity(files[key], config_dir, f"experiment {key}") for key in files
    }
    normalized = {
        **request,
        "repository": str(repository),
        "experiment_directory": str(experiment),
        "experiment_files": normalized_files,
        "rights_record": _identity(request["rights_record"], config_dir, "rights record"),
        "production_operations_record": _identity(
            request["production_operations_record"], config_dir, "production operations record"
        ),
        "deny_ledger": _identity(request["deny_ledger"], config_dir, "deny ledger"),
        "firewall_manifest": _identity(
            request["firewall_manifest"], config_dir, "firewall manifest"
        ),
        "tokenizer_decision": _identity(
            request["tokenizer_decision"], config_dir, "tokenizer decision"
        ),
        "packed_manifest": _identity(request["packed_manifest"], config_dir, "packed manifest"),
        "receipt_path": str(_path(request["receipt_path"], config_dir, "receipt path")),
    }
    return normalized


def _load_bound(identity, name):
    path = Path(identity["path"])
    if not path.is_file() or _sha256(path) != identity["sha256"]:
        raise ValueError(f"{name} identity mismatch")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _head(repository):
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def issue_launch_receipt(request, *, config_dir=None):
    """Issue one exclusive receipt only after every external authority cross-reference passes."""

    request = validate_launch_request(request, config_dir=config_dir)
    repository = Path(request["repository"])
    if _head(repository) != request["git_commit"]:
        raise ValueError("launch request Git commit is not checked out")
    experiment = Path(request["experiment_directory"])
    for name, identity in request["experiment_files"].items():
        path = Path(identity["path"])
        if path.parent != experiment or path.name != f"{name}.json":
            raise ValueError(
                f"experiment {name} identity is outside the exact experiment directory"
            )
        _load_bound(identity, f"experiment {name}")
    rights = _load_bound(request["rights_record"], "rights record")
    if (
        rights.get("format") != "speck_human_source_rights_acceptance"
        or rights.get("status") != "all_sources_human_approved"
        or rights.get("automated_approval_made") is not False
        or not rights.get("approved_source_ids")
    ):
        raise ValueError("human rights acceptance is incomplete")
    deny = _load_bound(request["deny_ledger"], "deny ledger")
    if (
        deny.get("format") != "speck_removal_deny_ledger"
        or deny.get("status") != "human_reviewed_deny_entries"
    ):
        raise ValueError("deny ledger is not human reviewed")
    operations = _load_bound(request["production_operations_record"], "operations record")
    required_operations = {
        "global_exact_deduplication",
        "global_near_deduplication",
        "acquisition_cleanup",
        "interruption_resume",
        "firewall_training_disjointness",
    }
    if (
        operations.get("format") != "speck_production_data_operations_qualification"
        or operations.get("status") != "production_data_operations_qualified"
        or any(operations.get("gates", {}).get(gate) != "pass" for gate in required_operations)
        or operations.get("deny_ledger_sha256") != request["deny_ledger"]["sha256"]
    ):
        raise ValueError("production operations authority is incomplete")
    firewall = _load_bound(request["firewall_manifest"], "firewall manifest")
    if (
        firewall.get("format") != "speck_data_firewall_manifest"
        or firewall.get("status") != "production_firewall_complete_rights_and_operations_bound"
        or firewall.get("authority", {}).get("rights_record_sha256")
        != request["rights_record"]["sha256"]
        or firewall.get("authority", {}).get("production_record_sha256")
        != request["production_operations_record"]["sha256"]
        or firewall.get("gates", {}).get("global_content_disjointness") != "pass"
        or firewall.get("gates", {}).get("sealed_audits_unopened") != "pass"
    ):
        raise ValueError("production firewall authority is incomplete")
    tokenizer = _load_bound(request["tokenizer_decision"], "tokenizer decision")
    if (
        tokenizer.get("format") != "speck_tokenizer_decision"
        or tokenizer.get("status") != "tokenizer_selected_and_frozen"
        or not isinstance(tokenizer.get("tokenizer_fingerprint"), str)
        or not tokenizer["tokenizer_fingerprint"]
    ):
        raise ValueError("tokenizer decision is incomplete")
    packed = _load_bound(request["packed_manifest"], "packed manifest")
    if (
        packed.get("format") != "speck_packed_tokens"
        or packed.get("tokenizer", {}).get("fingerprint") != tokenizer["tokenizer_fingerprint"]
        or packed.get("dedup", {}).get("scope") != "global"
    ):
        raise ValueError("packed dataset does not match the selected tokenizer or global dedup")
    packed_fingerprint = manifest_fingerprint(packed)
    receipt = {
        "format": RECEIPT_FORMAT,
        "format_version": FORMAT_VERSION,
        "status": "flagship_data_launch_preflight_passed",
        "git_commit": request["git_commit"],
        "repository": request["repository"],
        "experiment_directory": request["experiment_directory"],
        "experiment_files": request["experiment_files"],
        "authority_records": {
            "rights_record": request["rights_record"],
            "production_operations_record": request["production_operations_record"],
            "deny_ledger": request["deny_ledger"],
            "firewall_manifest": request["firewall_manifest"],
            "tokenizer_decision": request["tokenizer_decision"],
            "packed_manifest": request["packed_manifest"],
        },
        "rights_record_sha256": request["rights_record"]["sha256"],
        "production_operations_record_sha256": request["production_operations_record"]["sha256"],
        "deny_ledger_sha256": request["deny_ledger"]["sha256"],
        "firewall_manifest_sha256": request["firewall_manifest"]["sha256"],
        "tokenizer_decision_sha256": request["tokenizer_decision"]["sha256"],
        "tokenizer_fingerprint": tokenizer["tokenizer_fingerprint"],
        "packed_manifest": request["packed_manifest"]["path"],
        "packed_manifest_sha256": request["packed_manifest"]["sha256"],
        "packed_manifest_fingerprint": packed_fingerprint,
        "issued_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_authority": "data_preflight_only_explicit_marked_launch_required",
    }
    receipt_path = Path(request["receipt_path"])
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return receipt


def verify_launch_receipt(
    receipt_path,
    *,
    repository,
    experiment_directory,
    packed_manifest_fingerprint,
    tokenizer_fingerprint,
):
    """Verify the immutable receipt immediately before model construction."""

    path = Path(receipt_path).resolve()
    if not path.is_file():
        raise ValueError("required data launch receipt is missing")
    receipt = json.loads(path.read_text())
    if (
        receipt.get("format") != RECEIPT_FORMAT
        or receipt.get("format_version") != FORMAT_VERSION
        or receipt.get("status") != "flagship_data_launch_preflight_passed"
        or receipt.get("repository") != str(Path(repository).resolve())
        or receipt.get("experiment_directory") != str(Path(experiment_directory).resolve())
        or receipt.get("git_commit") != _head(Path(repository).resolve())
        or receipt.get("packed_manifest_fingerprint") != packed_manifest_fingerprint
        or receipt.get("tokenizer_fingerprint") != tokenizer_fingerprint
    ):
        raise ValueError("data launch receipt does not match this exact training launch")
    for identity in receipt["experiment_files"].values():
        if not Path(identity["path"]).is_file() or _sha256(identity["path"]) != identity["sha256"]:
            raise ValueError("experiment files changed after data preflight")
    records = receipt.get("authority_records")
    if not isinstance(records, dict) or set(records) != {
        "rights_record",
        "production_operations_record",
        "deny_ledger",
        "firewall_manifest",
        "tokenizer_decision",
        "packed_manifest",
    }:
        raise ValueError("data launch receipt lacks external authority records")
    rights = _load_bound(records["rights_record"], "receipt rights record")
    operations = _load_bound(
        records["production_operations_record"], "receipt production operations record"
    )
    deny = _load_bound(records["deny_ledger"], "receipt deny ledger")
    firewall = _load_bound(records["firewall_manifest"], "receipt firewall manifest")
    tokenizer = _load_bound(records["tokenizer_decision"], "receipt tokenizer decision")
    packed = _load_bound(records["packed_manifest"], "receipt packed manifest")
    if (
        rights.get("status") != "all_sources_human_approved"
        or rights.get("automated_approval_made") is not False
        or operations.get("status") != "production_data_operations_qualified"
        or deny.get("status") != "human_reviewed_deny_entries"
        or firewall.get("status") != "production_firewall_complete_rights_and_operations_bound"
        or tokenizer.get("status") != "tokenizer_selected_and_frozen"
        or manifest_fingerprint(packed) != packed_manifest_fingerprint
        or packed.get("tokenizer", {}).get("fingerprint") != tokenizer_fingerprint
    ):
        raise ValueError("external data authority changed or became invalid after preflight")
    return receipt
