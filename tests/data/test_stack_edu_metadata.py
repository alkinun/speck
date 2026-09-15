import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import speck.data.stack_edu_metadata as metadata
from speck.data.configuration import _validate_source
from speck.provenance.io import file_sha256


def make_unit(tmp_path, language="Rust", values=None):
    path = tmp_path / f"{language}.parquet"
    values = values or [language, language]
    rows = [
        {
            "blob_id": "a" * 40,
            "language": value,
            "repo_name": "example/repo",
            "path": "src/main",
            "src_encoding": "UTF-8",
            "length_bytes": 100,
            "score": 4.5,
            "int_score": 4,
            "detected_licenses": ["MIT"],
            "license_type": "permissive",
        }
        for value in values
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)
    reader = _validate_source(
        {
            "id": "stack_edu",
            "repo": "fixture/stack-edu",
            "revision": "a" * 40,
            "tree_path": language,
            "file_format": "parquet",
            "content_column": "blob_id",
            "filters": {},
        }
    )
    unit = {
        "id": language,
        "language": language,
        "reader": reader,
        "expected_file_rows": len(rows),
        "raw": {
            "source_id": "stack_edu",
            "filename": f"{language}/train-00000-of-00001.parquet",
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        },
    }
    return unit, path


def plan(tmp_path, units):
    return {
        "output_directory": str(tmp_path / "work"),
        "raw_directory": str(tmp_path / "raw"),
        "plan": {"path": str(tmp_path / "plan.json"), "sha256": "a" * 64},
        "inputs": {},
        "units": units,
    }


def test_language_check_covers_rows_beyond_the_first_and_rejects_nulls(tmp_path):
    for values in [["Rust", "Go"], ["Rust", None]]:
        unit, path = make_unit(tmp_path, values=values)
        with pytest.raises(ValueError, match="unexpected or missing language"):
            metadata.verify_metadata_file(path, unit)


def test_metadata_row_count_and_payload_identity_are_required(tmp_path):
    unit, path = make_unit(tmp_path)
    unit["expected_file_rows"] += 1
    with pytest.raises(ValueError, match="row count/schema"):
        metadata.verify_metadata_file(path, unit)
    unit["expected_file_rows"] -= 1
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="identity"):
        metadata.verify_metadata_file(path, unit)


def test_interrupted_metadata_acquisition_reuses_only_verified_published_units(
    tmp_path, monkeypatch
):
    first, a = make_unit(tmp_path)
    second, b = make_unit(tmp_path, "Go")
    spec = plan(tmp_path, [first, second])
    files = {"Rust": a, "Go": b}

    def raw_file(spec, unit):
        if unit["id"] == "Go":
            raise RuntimeError("injected download failure")
        return {
            "path": str(files[unit["id"]]),
            "bytes": unit["raw"]["bytes"],
            "sha256": unit["raw"]["sha256"],
        }

    monkeypatch.setattr(metadata, "_raw_file", raw_file)
    with pytest.raises(RuntimeError, match="injected"):
        metadata.acquire_metadata(spec, revision="a" * 40)
    first_receipt = (tmp_path / "work/Rust.json").read_bytes()
    with pytest.raises(ValueError, match="frozen execution"):
        metadata.acquire_metadata(spec, revision="b" * 40, resume=True)
    calls = []

    def resumed_raw(spec, unit):
        calls.append(unit["id"])
        return {
            "path": str(files[unit["id"]]),
            "bytes": unit["raw"]["bytes"],
            "sha256": unit["raw"]["sha256"],
        }

    monkeypatch.setattr(metadata, "_raw_file", resumed_raw)
    result = metadata.acquire_metadata(spec, revision="a" * 40, resume=True)
    assert calls == ["Go"]
    assert (tmp_path / "work/Rust.json").read_bytes() == first_receipt
    assert result["physical_rows"] == 4
    a.write_bytes(a.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="identity"):
        metadata.acquire_metadata(spec, revision="a" * 40, resume=True)


def test_verified_local_copy_preserves_original_metadata(tmp_path):
    unit, path = make_unit(tmp_path)
    original = path.read_bytes()
    spec = plan(tmp_path, [unit])
    spec["reuse_local_files"] = {"Rust": {"path": str(path), "sha256": file_sha256(path)}}
    result = metadata.acquire_metadata(spec, revision="a" * 40)
    assert Path(result["files"][0]["raw"]["path"]).read_bytes() == original
    assert path.read_bytes() == original
    assert result["files"][0]["raw"]["downloaded_this_invocation"] is False


def test_real_metadata_plan_binds_all_eleven_matched_languages():
    root = Path(__file__).resolve().parents[2]
    spec = metadata.load_metadata_plan(
        root / "research/flagship/stack_edu_metadata_acquisition_v1.json"
    )
    assert len(spec["units"]) == 11
    assert sum(unit["expected_file_rows"] for unit in spec["units"]) == 38669178
    assert {unit["language"] for unit in spec["units"]} == set(
        json.loads((root / "research/flagship/code_language_preparation_v1.json").read_text())[
            "language_weights_percent"
        ]
    )


def test_metadata_successor_keeps_original_units_and_only_adds_short_languages():
    root = Path(__file__).resolve().parents[2] / "research/flagship"
    original = metadata.load_metadata_plan(root / "stack_edu_metadata_acquisition_v1.json")
    successor = metadata.load_metadata_plan(root / "stack_edu_metadata_acquisition_v2.json")
    assert successor["units"][:11] == original["units"]
    assert len(successor["units"]) == 26
    assert len({unit["id"] for unit in successor["units"]}) == 26
    assert {unit["language"] for unit in successor["units"][11:]} == {
        "C++",
        "Java",
        "JavaScript",
        "Python",
    }
    assert successor["raw_directory"] == original["raw_directory"]
    assert successor["output_directory"] != original["output_directory"]


@pytest.mark.parametrize("version", [2, 3])
def test_metadata_successor_rejects_overlapping_output_and_changed_prefix(tmp_path, version):
    root = Path(__file__).resolve().parents[2] / "research/flagship"
    value = json.loads((root / f"stack_edu_metadata_acquisition_v{version}.json").read_text())
    for entry in value.values():
        if isinstance(entry, dict) and set(entry) == {"path", "sha256"}:
            entry["path"] = str((root / entry["path"]).resolve())
    path = tmp_path / "plan.json"
    original_output = value["output_directory"]
    value["output_directory"] = "/mnt/speck-data/speck/stack-edu-metadata-v1/work/nested"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="separate output"):
        metadata.load_metadata_plan(path)
    value["output_directory"] = original_output
    manifest = json.loads(Path(value["metadata_manifest"]["path"]).read_text())
    manifest["files"][0]["rows"] -= 1
    changed = tmp_path / "manifest.json"
    changed.write_text(json.dumps(manifest))
    value["metadata_manifest"] = {"path": str(changed), "sha256": file_sha256(changed)}
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="original prefix"):
        metadata.load_metadata_plan(path)


def test_third_metadata_view_preserves_all_verified_units_and_ancestor_outputs():
    root = Path(__file__).resolve().parents[2] / "research/flagship"
    previous = metadata.load_metadata_plan(root / "stack_edu_metadata_acquisition_v2.json")
    expanded = metadata.load_metadata_plan(root / "stack_edu_metadata_acquisition_v3.json")
    assert expanded["units"][:26] == previous["units"]
    assert [unit["raw"]["filename"] for unit in expanded["units"][26:]] == [
        "JavaScript/train-00002-of-00003.parquet",
        "Python/train-00002-of-00005.parquet",
    ]
    assert sum(unit["raw"]["bytes"] for unit in expanded["units"][26:]) == 929134037
    assert sum(unit["expected_file_rows"] for unit in expanded["units"]) == 110704408
    assert expanded["output_lineage"][:-1] == previous["output_lineage"]
    assert len(expanded["output_lineage"]) == 3
