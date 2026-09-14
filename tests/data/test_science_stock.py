import json
from pathlib import Path

import pyarrow as pa
import pytest

from speck.data.acquisition import iter_source_file_documents
from speck.data.acquisition_units import acquire_unit
from speck.data.configuration import _validate_source
from speck.data.production_rehearsal import _raw_local_path
from speck.data.science_stock import science_rejection
from speck.provenance.io import file_sha256
from tests.data.test_acquisition_units import fixture


def zstd_fixture(tmp_path, monkeypatch):
    plan, unit = fixture(tmp_path, monkeypatch)
    reader = _validate_source(
        {
            **unit["reader"],
            "file_format": "jsonl_zstd",
            "metadata_columns": {
                "id": "id",
                "url": "metadata.oa_url",
                "license": "metadata.oa_license",
            },
        }
    )
    path = _raw_local_path(Path(plan["raw_directory"]), reader, reader["revision"], "sample.zst")
    rows = [
        {
            "id": str(i),
            "text": f"Scientific paragraph {i} with distinct language and useful scientific content.",
            "score": 1 if i == 1 else 5,
            "metadata": {"oa_url": f"https://example.org/{i}", "oa_license": "CCBY"},
        }
        for i in range(6)
    ]
    with pa.output_stream(str(path), compression="zstd") as handle:
        handle.write("".join(json.dumps(row) + "\n" for row in rows).encode())
    unit.update(
        reader=reader,
        raw={
            "source_id": "example",
            "filename": "sample.zst",
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        },
        expected_file_rows=6,
    )
    return plan, unit


def test_zstd_preserves_nested_identity_and_physical_windows(tmp_path, monkeypatch):
    plan, unit = zstd_fixture(tmp_path, monkeypatch)
    docs = list(
        iter_source_file_documents(
            source=unit["reader"],
            revision=unit["reader"]["revision"],
            filename="sample.zst",
            filtering=plan["base"]["filtering"],
            cache_dir=plan["raw_directory"],
            keep_raw=True,
            start_row=1,
            stop_row=4,
        )
    )
    assert [doc["row"] for doc in docs] == [2, 3]
    assert docs[0]["metadata"] == {"id": "2", "url": "https://example.org/2", "license": "CCBY"}


def test_zstd_acquisition_resume_matches_clean_with_nested_metadata(tmp_path, monkeypatch):
    plan, unit = zstd_fixture(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="injected"):
        acquire_unit(plan, unit, tmp_path / "resumed", {}, interrupt_after_rows=3)
    recovered = acquire_unit(plan, unit, tmp_path / "resumed", {})
    clean = acquire_unit(plan, unit, tmp_path / "clean", {})
    assert recovered["manifest"]["output"] == clean["manifest"]["output"]
    rows = [
        json.loads(line)
        for line in (tmp_path / "clean" / unit["id"] / "records.jsonl").read_text().splitlines()
    ]
    assert rows[0]["url"] == "https://example.org/0"


def test_science_filter_requires_document_license_and_english(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    filters = json.loads(
        (root / "archive/pregrant-history/research/flagship/science_pes2o_v3_v1.json").read_text()
    )["filters"]
    text = (
        "This research studies biological processes using controlled experiments and numerical models. "
        * 8
    )
    document = {"content": text, "metadata": {"license": "CCBY"}}

    class Detector:
        def classify(self, text):
            return "en", 0.99

    monkeypatch.setattr("speck.data.science_stock._language_identifier", lambda: Detector())
    assert science_rejection(document, filters) == (None, 0.99)
    document["metadata"]["license"] = "CCBYNC"
    assert science_rejection(document, filters)[0] == "science_document_license"
    document["metadata"] = {}
    assert science_rejection(document, filters)[0] == "science_document_license"
