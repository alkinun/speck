import copy
import json

import pytest

from speck.data.acquisition_units import acquire_unit
from speck.data.admitted_math import load_math_preparation
from speck.data.checkpoint_replay import restore_reference_checkpoint
from speck.data.firewall_integration import analyze_exclusion, run_exclusion
from speck.data.rights import load_source_use_extension
from speck.provenance.io import file_sha256
from speck.provenance.repository import repository_root
from tests.data.test_acquisition_units import fixture as acquisition_fixture
from tests.data.test_firewall_integration import record
from tests.data.test_sqlite_wal import parent_fixture

ROOT = repository_root(__file__)
EXTENSION = ROOT / "research/flagship/ultradata_math_l2_source_use_v1.json"


def test_explicit_extension_admits_only_the_new_source_without_changing_parent():
    extension = load_source_use_extension(EXTENSION)
    assert extension["source"]["id"] == "ultradata_math_l2_preview"
    assert extension["decision"] == "approve"
    assert extension["automated_approval_made"] is False
    assert extension["training_authority"] is False
    plan = load_math_preparation(ROOT / "research/flagship/ultradata_math_l2_preparation_v1.json")
    assert len(plan["units"]) == 2
    assert plan["base"]["source_use_extension"]["sha256"] == file_sha256(EXTENSION)
    assert all(unit["category"] == "math" and unit["stop_row"] == 100000 for unit in plan["units"])


@pytest.mark.parametrize("change", ["pending", "scope", "authority"])
def test_extension_cannot_make_or_expand_a_human_decision(tmp_path, change):
    value = json.loads(EXTENSION.read_text())
    value["parent_acceptance"]["path"] = str(EXTENSION.parent / value["parent_acceptance"]["path"])
    if change == "pending":
        value["decision"] = "pending"
    elif change == "scope":
        value["scope_details"]["source_data_redistribution"] = True
    else:
        value["authority"]["authority_type"] = "agent"
    path = tmp_path / "extension.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_source_use_extension(path)


def test_math_english_rule_and_admission_identity_are_checkpoint_bound(tmp_path, monkeypatch):
    plan, unit = acquisition_fixture(tmp_path, monkeypatch)
    unit["category"] = "math"
    plan["base"]["math_english"] = {
        "language_policy": "math_prose",
        "minimum_prose_alphabetic_characters": 80,
        "minimum_detected_English_probability": 0.8,
    }
    plan["base"]["source_use_extension"] = {"path": "fixture", "sha256": "a" * 64}
    monkeypatch.setattr(
        "speck.data.sources.math_sample._language_result", lambda *args: ("English", 0.95)
    )
    result = acquire_unit(plan, unit, tmp_path / "units", {})
    rows = [
        json.loads(line)
        for line in (tmp_path / "units" / unit["id"] / "records.jsonl").read_text().splitlines()
    ]
    assert rows and all(row["detected_English_probability"] == 0.95 for row in rows)
    plan["base"]["source_use_extension"]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="owner/config"):
        acquire_unit(plan, unit, tmp_path / "units", {})
    assert result["manifest"]["retained_records"] == len(rows)


def test_new_math_input_reuses_only_reference_state(tmp_path):
    parent = parent_fixture(tmp_path)
    original = copy.deepcopy(parent)
    inputs = {}
    for category in ("web", "code", "math", "synthetic", "science", "reference"):
        path = tmp_path / f"new-{category}.jsonl"
        path.write_text(json.dumps(record("new_math_example")) + "\n" if category == "math" else "")
        inputs[f"acquired_train__{category}"] = {"path": str(path), "sha256": file_sha256(path)}
    config, restoration = restore_reference_checkpoint(
        parent, tmp_path / "new-excluded", candidate_inputs=inputs
    )
    assert parent == original
    assert config["sources"][:12] == parent["exclusion"]["result"]["manifest"]["sources"][:12]
    assert restoration["candidate_input_rebinding"] == inputs
    run_exclusion(config)
    analysis = analyze_exclusion(tmp_path / "new-excluded", parent["analysis"]["references"])
    assert analysis["retained"]["math"]["records"] == 1
    assert all(
        value["records"] == 0
        for category, value in analysis["retained"].items()
        if category != "math"
    )
    assert analysis["exact_reference_overlap"] == 0
    with pytest.raises(ValueError, match="all six"):
        restore_reference_checkpoint(
            parent,
            tmp_path / "bad",
            candidate_inputs={"acquired_train__math": inputs["acquired_train__math"]},
        )
