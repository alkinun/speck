"""Bind a checked local SQLite policy into a new preparation configuration."""

import json
from pathlib import Path

from speck.data.production_data import load_preprocess_config, validate_preprocess_config
from speck.data.sqlite_settings import validate_sqlite_settings, verify_sqlite_runtime
from speck.provenance.io import file_sha256


def load_preparation_policy(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if value.get("format") != "speck_preparation_sqlite_policy" or value.get("format_version") != 1:
        raise ValueError("unsupported preparation SQLite policy")
    declaration = validate_sqlite_settings(value["sqlite"])
    identity = value["qualification"]
    result_path = (path.parent / identity["path"]).resolve()
    if file_sha256(result_path) != identity["sha256"]:
        raise ValueError("SQLite policy qualification identity mismatch")
    result = json.loads(result_path.read_text())
    analysis = result.get("analysis", {})
    if (
        result.get("format") != "speck_sqlite_wal_comparison_result"
        or analysis.get("recommendation") != "65536_pages_for_this_qualified_envelope"
        or not all(
            analysis.get(key) is True
            for key in ("parity_gate_pass", "space_gate_pass", "hard_crash_recovery_pass")
        )
        or declaration["wal_autocheckpoint_pages"] != 65536
    ):
        raise ValueError("SQLite policy has no passing bounded qualification")
    candidates = [run for run in result["runs"] if run["policy"]["autocheckpoint_pages"] == 65536]
    if len(candidates) != 2:
        raise ValueError("SQLite qualification is missing its two candidate measurements")
    for run in candidates:
        verify_sqlite_runtime(
            declaration, {**run["policy"]["settings"], "sqlite_version": result["sqlite_version"]}
        )
    return {
        "sqlite": declaration,
        "policy": {"path": str(path), "sha256": file_sha256(path)},
        "qualification": {"path": str(result_path), "sha256": identity["sha256"]},
        "scope": value["scope"],
    }


def bind_preparation_policy(config_path, policy_path, destination):
    config_path = Path(config_path).resolve()
    parent = load_preprocess_config(config_path)
    policy = load_preparation_policy(policy_path)
    destination = Path(destination).resolve()
    if (
        destination == Path(parent["output_directory"])
        or destination.exists()
        or destination.with_name(destination.name + ".building").exists()
    ):
        raise ValueError("bound SQLite policy requires a new preparation destination")
    config = {key: value for key, value in parent.items() if key != "plan_fingerprint"}
    config.update(format_version=2, sqlite=policy["sqlite"], output_directory=str(destination))
    normalized = validate_preprocess_config(config)
    return config, {
        "format": "speck_preparation_policy_binding",
        "format_version": 1,
        "parent_config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "policy": policy["policy"],
        "qualification": policy["qualification"],
        "plan_fingerprint": normalized["plan_fingerprint"],
        "scope": policy["scope"],
        "training_authority": False,
    }
