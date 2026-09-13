import json

import pytest

from speck.data.checkpoint_replay import restore_reference_checkpoint
from speck.data.firewall_integration import run_exclusion
from speck.data.production_data import preprocess_sources
from speck.provenance.io import file_sha256
from tests.data.test_firewall_integration import fixture as reference_fixture
from tests.data.test_production_data import _config


def test_timing_preserves_outputs_and_accounts_for_disjoint_phases(tmp_path):
    plain = preprocess_sources(_config(tmp_path, "plain"))
    timing = {}
    measured = preprocess_sources(_config(tmp_path, "measured"), timing=timing)
    for key in ("outputs", "removals", "counts"):
        assert measured["manifest"][key] == plain["manifest"][key]
    assert timing["status"] == "complete"
    assert sum(timing["phases"].values()) == pytest.approx(timing["total_seconds"])
    assert set(timing["phases"]) == {
        "setup",
        "input_verification",
        "resume_verification",
        "processing",
        "publication",
        "final_verification",
    }
    assert sum(source["records_seen"] for source in timing["sources"]) == 6
    for source in timing["sources"]:
        assert source["processing_seconds_excluding_checkpoints"] >= 0
        assert source["elapsed_seconds"] == pytest.approx(
            source["checkpoint_seconds"] + source["processing_seconds_excluding_checkpoints"]
        )
    assert timing["checkpoint_seconds_all_phases"] >= sum(
        s["checkpoint_seconds"] for s in timing["sources"]
    )


def test_reopen_and_interruption_cannot_masquerade_as_complete_processing(tmp_path):
    config = _config(tmp_path)
    incomplete = {}
    with pytest.raises(RuntimeError, match="injected"):
        preprocess_sources(config, crash_after_records=3, timing=incomplete)
    assert incomplete["status"] == "incomplete"
    preprocess_sources(config)
    reopened = {}
    preprocess_sources(config, timing=reopened)
    assert reopened["status"] == "reopened"
    assert reopened["sources"] == []
    assert "processing" not in reopened["phases"]


def test_reference_replay_preserves_parent_and_candidate_results(tmp_path):
    references, config, _ = reference_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="injected"):
        run_exclusion(config, crash_after_records=references["records"] + 1)
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_bytes((tmp_path / "excluded.building/state.json").read_bytes())
    original = run_exclusion(config)
    manifest = tmp_path / "excluded/manifest.json"
    parent = {
        "status": "complete_reference_exclusion_and_bank_handoff_pass",
        "analysis": {
            "parent_manifest": {"path": str(manifest), "sha256": file_sha256(manifest)},
            "references": references,
        },
        "interruption": {
            "retained_checkpoint": {"path": str(snapshot), "sha256": file_sha256(snapshot)}
        },
        "exclusion": original,
    }
    before = file_sha256(tmp_path / "excluded/near_duplicates.sqlite3")
    rebound, restoration = restore_reference_checkpoint(parent, tmp_path / "replay")
    assert restoration["changed_checkpoint_fields"] == ["contract"]
    original_state = json.loads(snapshot.read_text())
    replay_state = json.loads((tmp_path / "replay.building/state.json").read_text())
    assert original_state["contract"] != replay_state["contract"]
    assert {**original_state, "contract": replay_state["contract"]} == replay_state
    timing = {}
    replay = run_exclusion(rebound, timing=timing)
    for key in ("outputs", "removals", "counts"):
        assert replay["result"]["manifest"][key] == original["result"]["manifest"][key]
    assert file_sha256(tmp_path / "excluded/near_duplicates.sqlite3") == before
    assert timing["status"] == "complete"
    assert not any(source["id"].startswith("firewall_reference__") for source in timing["sources"])
    with pytest.raises(FileExistsError):
        restore_reference_checkpoint(parent, tmp_path / "replay")
