"""Run a bounded production-data calibration through the six rehearsal stages."""

import json
from collections import Counter
from pathlib import Path

from speck.data_rehearsal import REQUIRED_GATES, REQUIRED_METRICS, STAGE_FORMAT
from speck.io import atomic_json, file_sha256
from speck.production_rehearsal import (
    STAGES,
    _fingerprint,
    run_production_rehearsal_stage,
    validate_production_rehearsal_plan,
)

FORMAT = "speck_production_calibration_plan"
FORMAT_VERSION = 1


def load_production_calibration_plan(path):
    """Validate a 2B calibration by scaling only its quotas through the 20B validator."""

    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if (
        value.get("format") != FORMAT
        or value.get("format_version") != FORMAT_VERSION
        or value.get("status") != "frozen_2B_calibration_not_operations_or_training_authority"
        or value.get("target_tokens") != 2_000_000_000
    ):
        raise ValueError("production calibration must be a frozen exact 2B plan")
    candidate = {
        **value,
        "format": "speck_production_rehearsal_plan",
        "status": "frozen_20B_rehearsal_not_training_authority",
        "target_tokens": 20_000_000_000,
        "sources": [
            {**source, "target_tokens": source["target_tokens"] * 10} for source in value["sources"]
        ],
    }
    normalized = validate_production_rehearsal_plan(candidate, config_dir=path.parent)
    normalized.update(
        {
            "format": FORMAT,
            "status": value["status"],
            "target_tokens": value["target_tokens"],
            "sources": [
                {**source, "target_tokens": source["target_tokens"]} for source in value["sources"]
            ],
        }
    )
    normalized.pop("plan_fingerprint")
    normalized["plan_fingerprint"] = _fingerprint(normalized)
    return normalized


def _load_stage_result(path, stage_id):
    path = Path(path)
    value = json.loads(path.read_text())
    if (
        value.get("format") != STAGE_FORMAT
        or value.get("format_version") != 1
        or value.get("status") != "complete"
        or value.get("stage_id") != stage_id
        or not isinstance(value.get("gates"), dict)
        or not isinstance(value.get("metrics"), dict)
    ):
        raise ValueError(f"production calibration stage result is invalid: {stage_id}")
    return value


def run_production_calibration(plan):
    """Run or resume all six stages and publish a non-authoritative scale projection."""

    output = Path(plan["output_directory"])
    stage_root = output.parent / "stage-results"
    stage_root.mkdir(parents=True, exist_ok=True)
    completed = []
    gates = {}
    metrics = {}
    for stage_id in STAGES:
        result_path = stage_root / f"{stage_id}.json"
        if result_path.exists():
            result = _load_stage_result(result_path, stage_id)
        else:
            result = run_production_rehearsal_stage(plan, stage_id, result_path)
        completed.append(
            {
                "id": stage_id,
                "result": {"path": str(result_path), "sha256": file_sha256(result_path)},
            }
        )
        gates.update(result["gates"])
        metrics.update(result["metrics"])
    failed = [gate for gate in REQUIRED_GATES if gates.get(gate) != "pass"]
    missing = [metric for metric in REQUIRED_METRICS if metric not in metrics]
    if failed or missing or metrics.get("packed_tokens", 0) < plan["target_tokens"]:
        raise RuntimeError(
            f"production calibration is incomplete: failed_gates={failed}, missing_metrics={missing}"
        )
    scale = 20_000_000_000 / plan["target_tokens"]
    projections = {
        "target_tokens": 20_000_000_000,
        "linear_scale_factor": scale,
        "projected_download_bytes": int(metrics["download_bytes"] * scale),
        "projected_filtered_bytes": int(metrics["filtered_bytes"] * scale),
        "projected_sqlite_index_bytes": int(metrics["sqlite_index_bytes"] * scale),
        "projected_packed_bytes": int(metrics["packed_bytes"] * scale),
        "measured_peak_rss_bytes": metrics["peak_rss_bytes"],
        "caveat": "byte and elapsed work are projected linearly from 2B; peak memory and source-specific throughput require confirmation during final-corpus preparation",
    }
    manifest = {
        "format": "speck_production_data_scale_calibration",
        "format_version": 1,
        "status": "production_2B_calibration_complete_not_20B_operations_or_training_authority",
        "plan_fingerprint": plan["plan_fingerprint"],
        "target_tokens": plan["target_tokens"],
        "rights_record": plan["rights_record"],
        "deny_ledger": plan["deny_ledger"],
        "stages": completed,
        "gates": gates,
        "metrics": metrics,
        "projection_20B": projections,
        "source_counts": dict(
            sorted(Counter(source["category"] for source in plan["sources"]).items())
        ),
        "operations_authority": False,
        "training_authority": False,
    }
    manifest_path = output.parent / "calibration-manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing != manifest:
            raise ValueError("published production calibration manifest changed")
    else:
        atomic_json(manifest_path, manifest)
    return manifest
