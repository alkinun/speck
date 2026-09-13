import gzip
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import speck.data.acquisition_units as units
from speck.data.acquisition import iter_source_file_documents
from speck.data.configuration import _validate_source
from speck.data.production_rehearsal import _raw_local_path
from speck.provenance.io import file_sha256
from tests.data.test_production_data import _config


def fixture(tmp_path, monkeypatch, file_format="parquet"):
    reader = _validate_source(
        {
            "id": "example",
            "repo": "fixture/example",
            "revision": "a" * 40,
            "tree_path": "",
            "content_column": "text",
            "file_format": file_format,
            "score_column": "score",
            "metadata_columns": {"id": "id"},
            "filters": {"min_score": 4},
        }
    )
    raw = tmp_path / "raw"
    raw.mkdir()
    filename = "sample.parquet" if file_format == "parquet" else "sample.json.gz"
    path = _raw_local_path(raw, reader, reader["revision"], filename)
    texts = [
        "First useful technical document with many distinct words about numerical algorithms.",
        "Filtered out by its source score.",
        "Second independent source record discussing biology and clinical research methods.",
        "A rejected security example with otherwise suitable distinct technical words.",
        "First useful technical document with many distinct words about numerical algorithms.",
        "Another paragraph describing historical libraries and ancient educational materials.",
    ]
    rows = [
        {"id": str(index), "text": text, "score": 1 if index == 1 else 5}
        for index, text in enumerate(texts)
    ]
    if file_format == "parquet":
        pq.write_table(pa.Table.from_pylist(rows), path)
    else:
        with gzip.open(path, "wt") as handle:
            handle.write("".join(json.dumps(row) + "\n" for row in rows))
    raw_identity = {
        "source_id": reader["id"],
        "filename": filename,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }
    unit = {
        "id": "example__rows_0_6",
        "category": "web",
        "reader": reader,
        "raw": raw_identity,
        "start_row": 0,
        "stop_row": 6,
    }
    config = _config(tmp_path)
    plan = {
        "raw_directory": str(raw),
        "checkpoint_rows": 2,
        "dedup_checkpoint_records": 2,
        "units": [unit],
        "base": {
            "filtering": {"min_chars": 1, "max_chars": 1000},
            "security": {"gitleaks_binary": {"path": "fixture"}},
            "contamination_plan": {"path": "fixture", "sha256": "a" * 64},
            "deny_ledger": config["deny_ledger"],
            "deduplication": config["policy"],
        },
    }
    monkeypatch.setattr(units, "_contamination_indexes", lambda _: {})
    monkeypatch.setattr(
        units,
        "_document_rejection",
        lambda doc, *_: "fixture_security" if doc["row"] == 3 else None,
    )

    def security_filter(path, binary, reports):
        reports.mkdir(parents=True, exist_ok=True)
        report = reports / "scan.json"
        report.write_text("[]")
        return 0, {"path": str(report), "sha256": file_sha256(report), "findings": 0}

    monkeypatch.setattr(units, "_gitleaks_filter", security_filter)
    return plan, unit


@pytest.mark.parametrize("file_format", ["parquet", "jsonl_gzip"])
def test_row_windows_use_original_rows_before_filters(tmp_path, monkeypatch, file_format):
    plan, unit = fixture(tmp_path, monkeypatch, file_format)
    rows = list(
        iter_source_file_documents(
            source=unit["reader"],
            revision=unit["reader"]["revision"],
            filename=unit["raw"]["filename"],
            filtering=plan["base"]["filtering"],
            cache_dir=plan["raw_directory"],
            keep_raw=True,
            start_row=1,
            stop_row=4,
        )
    )
    assert [row["row"] for row in rows] == [2, 3]
    assert rows[0]["metadata"]["id"] == "2"


@pytest.mark.parametrize("start,stop", [(-1, 3), (0, 0), (True, 3), (2, 1)])
def test_bad_windows_fail_before_download(tmp_path, monkeypatch, start, stop):
    plan, unit = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="row"):
        list(
            iter_source_file_documents(
                source=unit["reader"],
                revision=unit["reader"]["revision"],
                filename=unit["raw"]["filename"],
                filtering=plan["base"]["filtering"],
                cache_dir=plan["raw_directory"],
                keep_raw=True,
                start_row=start,
                stop_row=stop,
            )
        )


def test_unit_resume_and_metadata_match_uninterrupted(tmp_path, monkeypatch):
    plan, unit = fixture(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="injected"):
        units.acquire_unit(plan, unit, tmp_path / "resumed", {}, interrupt_after_rows=3)
    scratch = tmp_path / "resumed" / unit["id"] / "unscanned.jsonl"
    with scratch.open("ab") as handle:
        handle.write(b"torn tail")
    resumed = units.acquire_unit(plan, unit, tmp_path / "resumed", {})
    clean = units.acquire_unit(plan, unit, tmp_path / "clean", {})
    assert clean["manifest"] == resumed["manifest"]
    assert clean["manifest"]["yielded_rows"] == 5
    assert clean["manifest"]["retained_records"] == 4
    records = [
        json.loads(row)
        for row in (tmp_path / "clean" / unit["id"] / "records.jsonl").read_text().splitlines()
    ]
    assert [row["source_row"] for row in records] == [0, 2, 4, 5]
    assert records[1]["metadata"] == {"id": "2"}
    assert records[1]["source_revision"] == "a" * 40


def test_changed_state_or_raw_file_is_rejected(tmp_path, monkeypatch):
    plan, unit = fixture(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="injected"):
        units.acquire_unit(plan, unit, tmp_path / "resumed", {}, interrupt_after_rows=3)
    state_path = tmp_path / "resumed" / unit["id"] / "state.json"
    state = json.loads(state_path.read_text())
    state["state"]["next_row"] = 6
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="state identity"):
        units.acquire_unit(plan, unit, tmp_path / "resumed", {})
    path = _raw_local_path(
        Path(plan["raw_directory"]),
        unit["reader"],
        unit["reader"]["revision"],
        unit["raw"]["filename"],
    )
    path.write_bytes(b"corruption")
    with pytest.raises(ValueError, match="raw acquisition"):
        units.acquire_unit(plan, unit, tmp_path / "clean", {})


def test_ordered_cohort_dedup_recovery_and_replay_control(tmp_path, monkeypatch):
    plan, unit = fixture(tmp_path, monkeypatch)
    acquired = tmp_path / "acquired"
    units.acquire_unit(plan, unit, acquired, {})
    clean = units.deduplicate_units(plan, acquired, tmp_path / "clean-dedup")
    with pytest.raises(RuntimeError, match="injected"):
        units.deduplicate_units(plan, acquired, tmp_path / "resumed-dedup", crash_after_records=3)
    resumed = units.deduplicate_units(plan, acquired, tmp_path / "resumed-dedup")
    assert clean["result"]["manifest"]["counts"] == resumed["result"]["manifest"]["counts"]
    assert clean["result"]["manifest"]["outputs"] == resumed["result"]["manifest"]["outputs"]
    assert clean["result"]["manifest"]["outputs"]["exact_replay_control"]["bytes"] == 0
    assert clean["firewall_reference_exclusion"] == "not_run_on_this_engineering_cohort"
