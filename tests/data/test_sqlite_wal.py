import json
import subprocess
import sys
from pathlib import Path

import pytest

import speck.data.production_data as production_data
from speck.data.checkpoint_replay import restore_reference_checkpoint
from speck.data.firewall_integration import run_exclusion
from speck.data.sqlite_wal import CRASH_EXIT_CODE, assess_wal_comparison, wal_policy
from speck.provenance.io import file_sha256
from speck.provenance.repository import repository_root
from tests.data.test_firewall_integration import fixture as reference_fixture


def parent_fixture(tmp_path):
    references, config, _ = reference_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="injected"):
        run_exclusion(config, crash_after_records=references["records"] + 1)
    snapshot = tmp_path / "reference-state.json"
    snapshot.write_bytes((tmp_path / "excluded.building/state.json").read_bytes())
    completed = run_exclusion(config)
    manifest = tmp_path / "excluded/manifest.json"
    return {
        "status": "complete_reference_exclusion_and_bank_handoff_pass",
        "analysis": {
            "references": references,
            "parent_manifest": {"path": str(manifest), "sha256": file_sha256(manifest)},
        },
        "interruption": {
            "retained_checkpoint": {"path": str(snapshot), "sha256": file_sha256(snapshot)}
        },
        "exclusion": completed,
    }


@pytest.mark.parametrize("pages", [1000, 65536])
def test_policy_keeps_full_sync_and_restores_hooks_on_failure(tmp_path, pages):
    original = production_data._database
    checkpoint = production_data._checkpoint
    report = {}
    with pytest.raises(RuntimeError, match="fixture failure"):
        with wal_policy(pages, report):
            connection = production_data._database(tmp_path / "index.sqlite3")
            try:
                assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
                assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
                assert connection.execute("PRAGMA wal_autocheckpoint").fetchone()[0] == pages
            finally:
                connection.close()
            raise RuntimeError("fixture failure")
    assert production_data._database is original
    assert production_data._checkpoint is checkpoint
    assert report["nominal_trigger_bytes"] == pages * 4096


@pytest.mark.parametrize("pages", [0, -1, True, 1000.0, "1000", 262144])
def test_unfrozen_or_unbounded_policies_are_rejected(pages):
    with pytest.raises(ValueError, match="frozen"):
        with wal_policy(pages, {}):
            pass


@pytest.mark.parametrize("bound", [False, True])
def test_real_process_exit_recovers_committed_wal_and_discards_uncommitted_work(tmp_path, bound):
    parent = parent_fixture(tmp_path)
    declaration = (
        {
            "journal_mode": "WAL",
            "synchronous": "FULL",
            "wal_autocheckpoint_pages": 65536,
            "page_size": 4096,
            "cache_size_kib": 2000,
        }
        if bound
        else None
    )
    config, _ = restore_reference_checkpoint(
        parent, tmp_path / "crash", sqlite_settings=declaration
    )
    receipt_path = tmp_path / "receipt.json"
    spec = tmp_path / "worker.json"
    spec.write_text(json.dumps({"config": config, "pages": 65536, "receipt": str(receipt_path)}))
    child = subprocess.run(
        [sys.executable, "-m", "scripts.dedup_wal_compare", "--crash-worker", str(spec)],
        cwd=repository_root(__file__),
        check=False,
        capture_output=True,
        text=True,
    )
    assert child.returncode == CRASH_EXIT_CODE, child.stderr
    receipt = json.loads(receipt_path.read_text())
    assert receipt["main_file_documents_without_wal"] < receipt["committed_documents"]
    assert receipt["wal_bytes_at_crash"] > 32
    tail = receipt["uncommitted_tail"]
    assert Path(tail["path"]).stat().st_size == tail["committed_bytes"] + tail["bytes"]
    with wal_policy(65536, {}):
        recovered = run_exclusion(config)
    if bound:
        assert recovered["result"]["manifest"]["sqlite"] == declaration
    for key in ("outputs", "removals", "counts"):
        assert (
            recovered["result"]["manifest"][key] == parent["exclusion"]["result"]["manifest"][key]
        )


def assessment_inputs():
    plan = {
        "order": [
            {"id": name, "wal_autocheckpoint_pages": pages}
            for name, pages in (
                ("baseline_1", 1000),
                ("candidate_1", 65536),
                ("candidate_2", 65536),
                ("baseline_2", 1000),
            )
        ],
        "pairs": [["baseline_1", "candidate_1"], ["baseline_2", "candidate_2"]],
        "minimum_relative_reduction": 0.1,
        "maximum_observed_wal_bytes": 512,
    }
    runs = [
        {
            "id": item["id"],
            "timing": {
                "status": "complete",
                "total_seconds": 100 if item["id"].startswith("baseline") else 70,
            },
            "policy": {
                "autocheckpoint_pages": item["wal_autocheckpoint_pages"],
                "observed_peak_wal_bytes": 256,
            },
            "parity_pass": True,
        }
        for item in plan["order"]
    ]
    return plan, runs


def test_recommendation_requires_both_orders_space_and_recovery():
    plan, runs = assessment_inputs()
    assert assess_wal_comparison(plan, runs, True)["recommendation"].startswith("65536")
    assert assess_wal_comparison(plan, runs, False)["recommendation"].startswith("retain")
    runs[2]["timing"]["total_seconds"] = 95
    assert assess_wal_comparison(plan, runs, True)["recommendation"].startswith("retain")
    runs[2]["timing"]["total_seconds"] = 70
    runs[2]["policy"]["observed_peak_wal_bytes"] = 513
    assert assess_wal_comparison(plan, runs, True)["recommendation"].startswith("retain")


def test_final_flush_cost_cannot_be_hidden_by_lower_commit_time():
    plan, runs = assessment_inputs()
    runs[1]["timing"]["checkpoint_components_all_phases"] = {"sqlite_commit_seconds": 1}
    runs[1]["timing"]["total_seconds"] = 110
    assert not assess_wal_comparison(plan, runs, True)["pairs"][0]["speed_gate_pass"]
    runs[1]["timing"]["status"] = "reopened"
    with pytest.raises(ValueError, match="complete finite"):
        assess_wal_comparison(plan, runs, True)
