import json
from pathlib import Path

import pytest

import speck.data.source_stock as stock
from speck.data.acquisition_units import acquire_unit
from speck.provenance.io import file_sha256
from tests.data.test_acquisition_units import fixture
from tests.data.test_sqlite_wal import parent_fixture


def test_complete_shard_contract_rejects_wrong_physical_row_count(tmp_path, monkeypatch):
    plan, unit = fixture(tmp_path, monkeypatch)
    unit["expected_file_rows"] = 7
    with pytest.raises(ValueError, match="row count"):
        acquire_unit(plan, unit, tmp_path / "units", {})
    assert not (tmp_path / "units" / unit["id"] / "manifest.json").exists()


@pytest.mark.parametrize("category", ["math", "reference"])
def test_stock_runner_excludes_deduplicates_and_preserves_shortfall(
    tmp_path, monkeypatch, category
):
    raw_root = tmp_path / "source"
    raw_root.mkdir()
    plan, unit = fixture(raw_root, monkeypatch)
    unit.update(category=category, expected_file_rows=6)
    parent_root = tmp_path / "parent"
    parent_root.mkdir()
    parent = parent_fixture(parent_root)
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(json.dumps(parent))
    policy_path = tmp_path / "policy.json"
    declaration = {
        "journal_mode": "WAL",
        "synchronous": "FULL",
        "wal_autocheckpoint_pages": 65536,
        "page_size": 4096,
        "cache_size_kib": 2000,
    }
    policy_path.write_text(json.dumps({"sqlite": declaration}))
    monkeypatch.setattr(
        stock, "load_preparation_policy", lambda path: json.loads(Path(path).read_text())
    )
    monkeypatch.setattr(
        stock.subprocess,
        "check_output",
        lambda args, **kwargs: "" if "status" in args else "a" * 40,
    )
    # Token counting is separately exercised by real stock execution. This fixture
    # isolates orchestration with real acquisition, SQLite exclusion, and replay.
    monkeypatch.setattr(
        stock,
        "count_reference_tokens",
        lambda path, identity: {
            "documents": len(Path(path).read_text().splitlines()),
            "tokens": 15,
            "tokenizer": identity,
        },
    )
    plan.update(
        source_id="example",
        output_directory=str(tmp_path / "stock"),
        source_use={"identity": {"path": "fixture-approval", "sha256": "a" * 64}},
        reference_parent={"path": str(parent_path), "sha256": file_sha256(parent_path)},
        sqlite_policy={"path": str(policy_path), "sha256": file_sha256(policy_path)},
        reference_tokenizer={"path": "fixture-tokenizer", "sha256": "b" * 64},
        target_reference_tokens=100,
        maximum_observed_wal_bytes=2_147_483_648,
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    kwargs = dict(
        loader=lambda path: json.loads(path.read_text()),
        category=category,
        result_format="fixture_stock",
        boundary="fixture",
    )
    report = stock.prepare_source_stock(plan_path, tmp_path / "report.json", **kwargs)
    assert report["reference_capacity"]["documents"] == 3
    assert report["analysis"]["retained"][category]["records"] == 3
    assert report["analysis"]["exact_reference_overlap"] == 0
    assert report["capacity_target_pass"] is False
    assert report["status"] == "prepared_text_requires_capacity_or_storage_followup"
    assert report["training_authority"] is False
    assert "source_use" in report and "source_use_extension" not in report
    resumed = stock.prepare_source_stock(
        plan_path, tmp_path / "resumed.json", resume=True, **kwargs
    )
    assert resumed["analysis"] == report["analysis"]
    assert resumed["reference_capacity"] == report["reference_capacity"]
    with pytest.raises(FileExistsError):
        stock.prepare_source_stock(plan_path, tmp_path / "report.json", **kwargs)
