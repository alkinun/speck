import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from speck.data import stock_raw as raw
from speck.provenance.io import file_sha256


def fixture(tmp_path):
    path = tmp_path / "input.parquet"
    pq.write_table(pa.Table.from_pylist([{"text": "example", "id": "1"}]), path)
    unit = {
        "id": "first",
        "raw": {"bytes": path.stat().st_size, "sha256": file_sha256(path)},
        "expected_file_rows": 1,
        "reader": {"content_column": "text", "metadata_columns": {"id": "id"}},
    }
    return path, unit


def test_raw_resume_checks_published_bytes_and_retains_failure(tmp_path, monkeypatch):
    path, unit = fixture(tmp_path)
    plan = {
        "raw_acquisition_directory": str(tmp_path / "work"),
        "units": [unit, {**unit, "id": "second"}],
    }

    def fetch(plan, unit):
        if unit["id"] == "second":
            raise RuntimeError("interrupted")
        return {"path": str(path), **unit["raw"]}

    monkeypatch.setattr(raw, "_raw_file", fetch)
    with pytest.raises(RuntimeError):
        raw.acquire_stock_raw(plan, {"revision": "one"})
    receipt = (tmp_path / "work/first.json").read_bytes()
    assert len(list((tmp_path / "work").glob("*-failed-*.json"))) == 1
    calls = []

    def resumed(plan, unit):
        calls.append(unit["id"])
        return {"path": str(path), **unit["raw"]}

    monkeypatch.setattr(raw, "_raw_file", resumed)
    with pytest.raises(ValueError, match="same frozen"):
        raw.acquire_stock_raw(plan, {"revision": "two"}, resume=True)
    result = raw.acquire_stock_raw(plan, {"revision": "one"}, resume=True)
    assert calls == ["second"] and result["physical_rows"] == 2
    assert (tmp_path / "work/first.json").read_bytes() == receipt
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="identity"):
        raw.acquire_stock_raw(plan, {"revision": "one"}, resume=True)


def test_raw_requires_physical_schema_and_count(tmp_path):
    path, unit = fixture(tmp_path)
    unit["expected_file_rows"] = 2
    with pytest.raises(ValueError, match="schema/physical"):
        raw.verify_raw_parquet(path, unit)
    unit["expected_file_rows"] = 1
    unit["reader"]["metadata_columns"]["url"] = "missing"
    with pytest.raises(ValueError, match="schema/physical"):
        raw.verify_raw_parquet(path, unit)
