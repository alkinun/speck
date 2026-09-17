import pytest

from speck.data.production_data import preprocess_sources
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
    assert (
        sum(timing["checkpoint_components_all_phases"].values())
        <= timing["checkpoint_seconds_all_phases"]
    )
    for source in timing["sources"]:
        assert sum(source["checkpoint_components"].values()) <= source["checkpoint_seconds"]


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
