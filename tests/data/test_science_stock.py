import json
from pathlib import Path

import pyarrow as pa
import pytest

from speck.data.acquisition import iter_source_file_documents
from speck.data.acquisition_units import acquire_unit
from speck.data.configuration import _validate_source
from speck.data.production_rehearsal import _raw_local_path
from speck.data.science_stock import load_science_preparation, science_rejection
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
        language = "en"

        def classify(self, text):
            return self.language, 0.99

    detector = Detector()
    monkeypatch.setattr("speck.data.science_stock._language_identifier", lambda: detector)
    assert science_rejection(document, filters) == (None, 0.99)
    document["metadata"]["license"] = "CCBYNC"
    assert science_rejection(document, filters)[0] == "science_document_license"
    document["metadata"] = {}
    assert science_rejection(document, filters)[0] == "science_document_license"
    document["metadata"]["license"] = "CCBY"
    detector.language = "fr"
    assert science_rejection(document, filters)[0] == "science_non_English"


def test_headroom_successor_preserves_first_unit_and_binds_only_next_shard(tmp_path):
    root = Path(__file__).resolve().parents[2]
    original = root / "research/flagship/pes2o_stock_preparation_v1.json"
    value = json.loads(original.read_text())
    for key in (
        "base_plan",
        "source_qualification",
        "reference_parent",
        "sqlite_policy",
        "tokenizer_decision",
    ):
        value[key]["path"] = str((original.parent / value[key]["path"]).resolve())
    intake = {
        "format": "speck_science_complete_shard_intake",
        "format_version": 1,
        "training_authority": False,
        "repo": "allenai/peS2o",
        "revision": "636a503e44a3ca1b58e01fb61eab0825cd574de0",
        "filename": "data/v3/train-0041-of-0136.zst",
        "physical_rows": 110183,
        "raw": {
            "bytes": 998912463,
            "sha256": "c20829cbe5e28f8dab337537080cec1fdd82a8eabbeeaf85c6fa927cb1d959d6",
        },
    }
    intake_path = tmp_path / "intake.json"
    intake_path.write_text(json.dumps(intake))
    value.update(
        format_version=2,
        target_reference_tokens=480000000,
        additional_shard_intake={"path": str(intake_path), "sha256": file_sha256(intake_path)},
    )
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(value))
    plan = load_science_preparation(path)
    assert plan["units"][0] == load_science_preparation(original)["units"][0]
    assert plan["units"][1]["raw"]["filename"] == intake["filename"]
    assert (
        plan["base"]["science_filters"]
        == load_science_preparation(original)["base"]["science_filters"]
    )
