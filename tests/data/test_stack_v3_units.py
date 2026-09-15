import hashlib
import json
from copy import deepcopy

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import speck.data.stack_v3_units as units
from speck.data.acquisition_units import _unit_config
from speck.data.firewall_integration import group_acquisition_units
from speck.provenance.io import durable_json, file_sha256


def setup(tmp_path, monkeypatch):
    text = "x = 1\n" * 40
    file = {
        "content_id": "b" * 40,
        "content": text,
        "size_bytes": len(text),
        "language": "Python",
        "file_path": "src/example.py",
        "is_vendor": False,
        "license_type": "permissive",
        "detected_licenses": ["MIT"],
    }
    repository = {
        "repo_path": "owner/repo",
        "repo_id": "original-repo-id",
        "commit_id": "a" * 40,
        "github_metadata": {"is_fork": False},
        "files": [file, {**file, "file_path": "vendor/example.py"}],
    }
    raw = tmp_path / "input.parquet"
    pq.write_table(pa.Table.from_pylist([repository, repository]), raw, row_group_size=1)
    declaration = {
        "path": str(raw),
        "filename": "pinned.parquet",
        "bytes": raw.stat().st_size,
        "sha256": file_sha256(raw),
        "row_group_rows": [1, 1],
    }
    unit = {
        "id": "stack_v3__file_000__group_001",
        "category": "code",
        "raw": declaration,
        "row_group": 1,
        "start_row": 1,
        "stop_row": 2,
        "source": {"repo": "approved/source", "revision": "c" * 40},
    }
    policy = {
        "languages": ["Python"],
        "accepted_detected_licenses": ["MIT"],
        "min_file_bytes": 1,
        "max_file_bytes": 1000000,
        "excluded_path_components": ["vendor"],
        "English_prose": {"minimum_alphabetic_characters": 80, "minimum_probability": 0.8},
    }
    plan = {
        "units": [unit],
        "checkpoint_rows": 16,
        "reference_tokenizer": {"path": "fixture"},
        "base": {
            "stack_v3_policy": policy,
            "filtering": {"min_chars": 200, "max_chars": 100000},
            "security": {"gitleaks_binary": {"path": "fixture"}},
            "contamination_plan": {"fixture": True},
        },
    }
    monkeypatch.setattr(units, "content_rejection", lambda file, policy: None)
    monkeypatch.setattr(units, "_document_rejection", lambda *args: None)

    def scan(path, binary, reports):
        report = reports / "scan.json"
        durable_json(report, [])
        return 0, {"path": str(report), "sha256": file_sha256(report), "findings": 0}

    monkeypatch.setattr(units, "_gitleaks_filter", scan)

    def count(path, tokenizer):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        return {
            "documents": len(records),
            "tokens": len(records) * 42,
            "by_language": {"Python": {"documents": len(records), "tokens": len(records) * 42}},
        }

    monkeypatch.setattr(units, "count_code_tokens", count)
    return plan, unit, text


def test_interrupted_unit_preserved_replay_matches_clean_and_groups(tmp_path, monkeypatch):
    plan, unit, text = setup(tmp_path, monkeypatch)
    units.verify_raw(unit["raw"])
    output = tmp_path / "resumed"
    with pytest.raises(RuntimeError, match="injected"):
        units.acquire_stack_v3_unit(plan, unit, output, {}, interrupt_after_repositories=1)
    failed = output / unit["id"] / "attempt-00000"
    before = {str(p): p.read_bytes() for p in failed.rglob("*") if p.is_file()}
    result = units.acquire_stack_v3_unit(plan, unit, output, {})
    clean = units.acquire_stack_v3_unit(plan, unit, tmp_path / "clean", {})
    assert result["manifest"]["output"]["sha256"] == clean["manifest"]["output"]["sha256"]
    assert result["manifest"]["rejections"] == {"declared_vendor_path": 1, "gitleaks": 0}
    assert result["manifest"]["physical_files_seen"] == 2
    assert all(open(path, "rb").read() == data for path, data in before.items())
    record = json.loads((output / unit["id"] / result["manifest"]["output"]["path"]).read_text())
    assert record["source_row"] == 1 and record["source_file_index"] == 0
    assert record["commit_id"] == "a" * 40 and record["metadata"]["detected_licenses"] == ["MIT"]
    assert record["content_id"] == "b" * 40
    assert record["released_content_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert units.acquire_stack_v3_unit(plan, unit, output, {})["reused"]
    grouped = group_acquisition_units(plan, output, tmp_path / "groups")
    assert grouped["outputs"]["code"]["sha256"] == result["manifest"]["output"]["sha256"]


def test_reopen_rejects_changed_content_raw_policy_and_wrong_category(tmp_path, monkeypatch):
    plan, unit, _ = setup(tmp_path, monkeypatch)
    output = tmp_path / "output"
    result = units.acquire_stack_v3_unit(plan, unit, output, {})
    changed = deepcopy(plan)
    changed["base"]["stack_v3_policy"]["accepted_detected_licenses"].append("Apache-2.0")
    with pytest.raises(ValueError, match="configuration changed"):
        units.acquire_stack_v3_unit(changed, unit, output, {})
    with pytest.raises(ValueError, match="code units"):
        _unit_config(plan, {**unit, "category": "web"})
    final = output / unit["id"] / result["manifest"]["output"]["path"]
    final.write_text("changed")
    with pytest.raises(ValueError, match="payload changed"):
        units.acquire_stack_v3_unit(plan, unit, output, {})
    wrong = {**unit["raw"], "row_group_rows": [2]}
    with pytest.raises(ValueError, match="row-group"):
        units.verify_raw(wrong)


def test_output_bound_preserves_failure_without_publishing(tmp_path, monkeypatch):
    plan, unit, _ = setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="allowance exhausted"):
        units.acquire_stack_v3_unit(plan, unit, tmp_path / "output", {}, remaining_bytes=1)
    directory = tmp_path / "output" / unit["id"]
    assert not (directory / "manifest.json").exists()
    assert (
        json.loads((directory / "attempt-00000/result.json").read_text())["status"]
        == "failed_preserved"
    )
