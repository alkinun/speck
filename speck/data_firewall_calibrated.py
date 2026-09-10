"""Construct a production firewall from the accepted 2B operations fallback."""

import json
from pathlib import Path

import speck.data_firewall as base
from speck.io import file_sha256

FORMAT = "speck_calibrated_data_firewall"
FORMAT_VERSION = 1
FALLBACK_STATUS = "project_owner_accepted_non_gpu_operations_fallback_through_150B"


def _identity(value, config_dir, context):
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{context} must contain path and sha256")
    path = Path(value["path"]).expanduser()
    path = (config_dir / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_file() or file_sha256(path) != value["sha256"]:
        raise ValueError(f"{context} identity mismatch")
    return {"path": str(path), "sha256": value["sha256"]}


def validate_calibrated_firewall_config(config, *, config_dir=None):
    """Validate firewall inputs and bind rights plus the accepted calibration fallback."""

    config_dir = Path(config_dir or ".").resolve()
    expected = {
        "format",
        "format_version",
        "status",
        "authority",
        "policy",
        "categories",
        "output_directory",
    }
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError("calibrated firewall fields are invalid")
    if (
        config["format"] != FORMAT
        or config["format_version"] != FORMAT_VERSION
        or config["status"] != "construction_authorized_not_training_authority"
    ):
        raise ValueError("unsupported calibrated firewall contract")
    authority = config["authority"]
    if not isinstance(authority, dict) or set(authority) != {
        "rights_record",
        "operations_fallback",
    }:
        raise ValueError("calibrated firewall authority fields are invalid")
    rights = _identity(authority["rights_record"], config_dir, "rights record")
    fallback = _identity(authority["operations_fallback"], config_dir, "operations fallback")
    fixture = {
        "format": "speck_data_firewall",
        "format_version": 1,
        "status": config["status"],
        "authority": {
            "mode": "fixture_only",
            "rights_record": None,
            "rights_record_sha256": None,
            "rights_status": None,
            "production_record": None,
            "production_record_sha256": None,
            "production_status": None,
        },
        "policy": config["policy"],
        "categories": config["categories"],
        "output_directory": config["output_directory"],
    }
    normalized = base.validate_firewall_config(fixture, config_dir=config_dir)
    normalized["format"] = FORMAT
    normalized["authority"] = {
        "mode": "production",
        "rights_record": rights["path"],
        "rights_record_sha256": rights["sha256"],
        "rights_status": "all_sources_human_approved",
        "production_record": fallback["path"],
        "production_record_sha256": fallback["sha256"],
        "production_status": FALLBACK_STATUS,
    }
    normalized.pop("plan_fingerprint")
    normalized["plan_fingerprint"] = base._fingerprint(normalized)
    return normalized


def load_calibrated_firewall_config(path):
    path = Path(path).resolve()
    return validate_calibrated_firewall_config(json.loads(path.read_text()), config_dir=path.parent)


def _verify_calibrated_authority(config):
    authority = config["authority"]
    rights_path = Path(authority["rights_record"])
    fallback_path = Path(authority["production_record"])
    if file_sha256(rights_path) != authority["rights_record_sha256"]:
        raise ValueError("calibrated firewall rights identity mismatch")
    if file_sha256(fallback_path) != authority["production_record_sha256"]:
        raise ValueError("calibrated firewall fallback identity mismatch")
    rights = json.loads(rights_path.read_text())
    fallback = json.loads(fallback_path.read_text())
    if (
        rights.get("format") != "speck_human_source_rights_acceptance"
        or rights.get("status") != "all_sources_human_approved"
        or rights.get("automated_approval_made") is not False
    ):
        raise ValueError("calibrated firewall human rights acceptance is incomplete")
    declared = {
        item["source_id"] for category in config["categories"] for item in category["inputs"]
    }
    if not declared <= set(rights.get("approved_source_ids", ())):
        raise ValueError("calibrated firewall contains a source outside human approval")
    if (
        fallback.get("format") != "speck_production_data_calibration_fallback"
        or fallback.get("status") != FALLBACK_STATUS
        or fallback.get("training_authority") is not False
    ):
        raise ValueError("calibrated operations fallback is invalid")
    analysis_identity = fallback.get("calibration_analysis", {})
    analysis_path = Path(analysis_identity.get("path", ""))
    if not analysis_path.is_absolute():
        analysis_path = Path(__file__).parents[1] / analysis_path
    if not analysis_path.is_file() or file_sha256(analysis_path) != analysis_identity.get("sha256"):
        raise ValueError("calibrated operations analysis identity mismatch")
    analysis = json.loads(analysis_path.read_text())
    required = {
        "source_identity",
        "global_exact_deduplication",
        "global_near_deduplication",
        "packing_integrity",
        "acquisition_cleanup",
        "interruption_resume",
        "firewall_training_disjointness",
    }
    if any(analysis.get("gates", {}).get(gate) != "pass" for gate in required):
        raise ValueError("calibrated operations analysis lacks required gates")


def construct_calibrated_firewall(config, *, restart=False):
    """Use the frozen builder after replacing only its external-authority verifier."""

    if "plan_fingerprint" not in config:
        config = validate_calibrated_firewall_config(config)
    else:
        payload = {key: value for key, value in config.items() if key != "plan_fingerprint"}
        if config["plan_fingerprint"] != base._fingerprint(payload):
            raise ValueError("normalized calibrated firewall fingerprint mismatch")
    original = base._verify_authority
    base._verify_authority = _verify_calibrated_authority
    try:
        return base.construct_firewall(config, restart=restart)
    finally:
        base._verify_authority = original
