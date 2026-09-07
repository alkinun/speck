import hashlib
import json
import sys
from pathlib import Path

import pytest

from speck.data_rehearsal import (
    issue_operations_qualification,
    run_rehearsal,
    validate_rehearsal_config,
    verify_rehearsal_manifest,
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fixture(tmp_path, output="rehearsal"):
    script = tmp_path / "stage.py"
    script.write_text(
        """import json, os, pathlib, sys
stage, result, counter = sys.argv[1:]
counter = pathlib.Path(counter)
counter.write_text(str(int(counter.read_text()) + 1) if counter.exists() else '1')
gates = {
 'source_identity': {'source_identity': 'pass'},
 'acquisition': {},
 'global_dedup': {'global_exact_deduplication': 'pass', 'global_near_deduplication': 'pass'},
 'packing': {'packing_integrity': 'pass'},
 'resume_cleanup': {'acquisition_cleanup': 'pass', 'interruption_resume': 'pass'},
 'firewall_disjointness': {'firewall_training_disjointness': 'pass'},
}[stage]
metrics = {
 'source_identity': {'records_seen': 10},
 'acquisition': {'download_bytes': 1000, 'download_bytes_per_second': 100, 'filtered_bytes': 900},
 'global_dedup': {'records_retained': 8, 'sqlite_index_bytes': 200, 'peak_rss_bytes': 300},
 'packing': {'packing_tokens_per_second': 400, 'packed_tokens': int(os.environ.get('TARGET_TOKENS', '1000')), 'packed_bytes': 500, 'unique_tokens': 600},
 'resume_cleanup': {},
 'firewall_disjointness': {},
}[stage]
pathlib.Path(result).write_text(json.dumps({'format':'speck_data_rehearsal_stage_result','format_version':1,'status':'complete','stage_id':stage,'gates':gates,'metrics':metrics}))
"""
    )
    deny = tmp_path / "deny.json"
    deny.write_text(
        json.dumps(
            {
                "format": "speck_removal_deny_ledger",
                "format_version": 1,
                "status": "human_reviewed_deny_entries",
                "entries": [],
            }
        )
    )
    stages = []
    for stage_id in (
        "source_identity",
        "acquisition",
        "global_dedup",
        "packing",
        "resume_cleanup",
        "firewall_disjointness",
    ):
        result = tmp_path / f"{output}-{stage_id}.json"
        counter = tmp_path / f"{output}-{stage_id}.count"
        stages.append(
            {
                "id": stage_id,
                "command": [sys.executable, str(script), stage_id, str(result), str(counter)],
                "cwd": str(tmp_path),
                "environment": {},
                "inputs": [{"path": str(script), "sha256": _sha256(script)}],
                "result": {"path": str(result)},
            }
        )
    return {
        "format": "speck_data_rehearsal",
        "format_version": 1,
        "status": "orchestration_authorized_not_training_authority",
        "mode": "fixture_only",
        "target_tokens": 1000,
        "rights_record": None,
        "deny_ledger": {"path": str(deny), "sha256": _sha256(deny)},
        "stages": stages,
        "required_gates": [
            "source_identity",
            "global_exact_deduplication",
            "global_near_deduplication",
            "packing_integrity",
            "acquisition_cleanup",
            "interruption_resume",
            "firewall_training_disjointness",
        ],
        "required_metrics": [
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
        ],
        "output_directory": str(tmp_path / output),
    }


def test_fixture_rehearsal_captures_all_gates_metrics_logs_and_telemetry(tmp_path):
    config = validate_rehearsal_config(_fixture(tmp_path))
    manifest = run_rehearsal(config)

    assert manifest["status"] == "fixture_rehearsal_complete_not_production_authority"
    assert all(manifest["gates"][gate] == "pass" for gate in config["required_gates"])
    assert set(config["required_metrics"]) <= set(manifest["metrics"])
    assert len(manifest["stages"]) == 6
    assert all(stage["telemetry"]["elapsed_seconds"] >= 0 for stage in manifest["stages"])
    assert manifest["training_authority"] == "blocked"
    assert run_rehearsal(config) == manifest
    assert verify_rehearsal_manifest(Path(config["output_directory"]) / "manifest.json") == manifest
    with pytest.raises(ValueError, match="production 20B rehearsal"):
        issue_operations_qualification(
            Path(config["output_directory"]) / "manifest.json", tmp_path / "operations.json"
        )


def test_rehearsal_resume_skips_committed_stages_and_detects_changed_results(tmp_path):
    config = validate_rehearsal_config(_fixture(tmp_path, "resumed"))
    with pytest.raises(RuntimeError, match="injected rehearsal orchestrator crash"):
        run_rehearsal(config, crash_after_stage="global_dedup")
    manifest = run_rehearsal(config)

    for stage in manifest["stages"][:3]:
        counter = tmp_path / f"resumed-{stage['id']}.count"
        assert counter.read_text() == "1"

    changed = validate_rehearsal_config(_fixture(tmp_path, "changed"))
    with pytest.raises(RuntimeError, match="injected rehearsal orchestrator crash"):
        run_rehearsal(changed, crash_after_stage="acquisition")
    result = tmp_path / "changed-acquisition.json"
    result.write_text("{}")
    with pytest.raises(ValueError, match="result identity mismatch"):
        run_rehearsal(changed)


def test_production_rehearsal_requires_rights_and_exact_20B_target(tmp_path):
    raw = _fixture(tmp_path, "production")
    raw["mode"] = "production_20B"
    with pytest.raises(ValueError, match="exactly 20B"):
        validate_rehearsal_config(raw)


def test_complete_temporary_production_contract_can_issue_operations_record(tmp_path):
    raw = _fixture(tmp_path, "production-complete")
    rights = tmp_path / "rights.json"
    rights.write_text(
        json.dumps(
            {
                "format": "speck_human_source_rights_acceptance",
                "status": "all_sources_human_approved",
                "approved_source_ids": ["fixture"],
                "automated_approval_made": False,
            }
        )
    )
    raw["mode"] = "production_20B"
    raw["target_tokens"] = 20_000_000_000
    raw["rights_record"] = {"path": str(rights), "sha256": _sha256(rights)}
    next(stage for stage in raw["stages"] if stage["id"] == "packing")["environment"] = {
        "TARGET_TOKENS": "20000000000"
    }
    config = validate_rehearsal_config(raw)
    manifest = run_rehearsal(config)
    record = issue_operations_qualification(
        Path(config["output_directory"]) / "manifest.json", tmp_path / "operations.json"
    )

    assert manifest["target_tokens"] == 20_000_000_000
    assert record["status"] == "production_data_operations_qualified"
    assert all(value == "pass" for value in record["gates"].values())


def test_published_rehearsal_detects_changed_stage_logs(tmp_path):
    config = validate_rehearsal_config(_fixture(tmp_path, "published"))
    manifest = run_rehearsal(config)
    log = Path(config["output_directory"]) / manifest["stages"][0]["stdout"]["path"]
    log.write_text("changed")

    with pytest.raises(ValueError, match="stdout identity mismatch"):
        run_rehearsal(config)


def test_flagship_rehearsal_plan_keeps_real_20B_manifest_blocked():
    root = Path(__file__).parents[1]
    plan = json.loads((root / "research/flagship/data_rehearsal_plan.json").read_text())

    assert plan["status"] == "fixture_orchestration_ready_real_20B_manifest_blocked"
    assert plan["stage_order"] == list(
        (
            "source_identity",
            "acquisition",
            "global_dedup",
            "packing",
            "resume_cleanup",
            "firewall_disjointness",
        )
    )
    assert plan["production_mode"]["target_tokens"] == 20_000_000_000
    assert plan["production_mode"]["fixture_may_issue_operations_record"] is False
    assert plan["real_manifest"]["status"] == "blocked"
    assert plan["training_authority"] == "blocked"
