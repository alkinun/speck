import copy
import hashlib
import json

import pytest

from speck.data.firewall_integration import (
    CATEGORIES,
    analyze_exclusion,
    exclusion_config,
    group_acquisition_units,
    make_reference_controls,
    reference_sources,
    run_exclusion,
    verify_excluded_parent,
)
from speck.data.source_bank import load_bank_plan, prepare_source_bank
from speck.provenance.io import file_sha256
from tests.data.test_acquisition_units import fixture as acquisition_fixture
from tests.data.test_production_data import _config
from tests.data.test_source_bank import ByteTokenizer


def record(prefix):
    text = " ".join(f"{prefix}_{i}" for i in range(120))
    return {"text": text, "released_content_sha256": hashlib.sha256(text.encode()).hexdigest()}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": file_sha256(path)}


def fixture(tmp_path):
    policy_config = _config(tmp_path)
    outputs = {}
    for role in ("unseen", "primary"):
        for category in CATEGORIES:
            key = f"{category}_{role}"
            path = tmp_path / "references" / f"{key}.jsonl"
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(record(key)) + "\n")
            outputs[key] = {
                "path": path.name,
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
    dedup = write(
        tmp_path / "references/manifest.json",
        {"outputs": outputs, "policy": policy_config["policy"]},
    )
    prepared = write(
        tmp_path / "prepared.json",
        {
            "status": "views_and_global_near_dedup_complete_not_consumer_authority",
            "deduplication": {
                "outputs": outputs,
                "manifest": dedup,
                "counts": {"records_retained": 12},
            },
        },
    )
    firewall = write(
        tmp_path / "firewall.json",
        {
            "status": "production_firewall_complete_rights_and_operations_bound",
            "categories": [
                {
                    "id": category,
                    "inputs": [
                        {
                            "id": f"{category}_{role}",
                            "path": str(
                                tmp_path / "references" / outputs[f"{category}_{role}"]["path"]
                            ),
                            "sha256": outputs[f"{category}_{role}"]["sha256"],
                        }
                        for role in ("unseen", "primary")
                    ],
                    "outputs": {"D5_tokenizer": {"path": "MUST_NOT_OPEN", "sealed": True}},
                }
                for category in CATEGORIES
            ],
        },
    )
    identity = write(
        tmp_path / "firewall-plan.json",
        {
            "format": "speck_flagship_firewall_plan",
            "format_version": 4,
            "construction": {"runtime_manifest": firewall},
            "input_preparation": {"runtime_manifest": prepared},
        },
    )
    references = reference_sources(identity, tmp_path)
    groups = tmp_path / "groups"
    groups.mkdir()
    group_outputs = {}
    for category in CATEGORIES:
        path = groups / f"{category}.jsonl"
        path.write_text(
            json.dumps(record(f"training_{category}"))
            + "\n"
            + json.dumps(record(f"{category}_primary"))
            + "\n"
        )
        group_outputs[category] = {"path": path.name, "sha256": file_sha256(path)}
    write(groups / "manifest.json", {"outputs": group_outputs})
    plan = {
        "base": {
            "deduplication": policy_config["policy"],
            "deny_ledger": policy_config["deny_ledger"],
        }
    }
    controls = make_reference_controls(references, tmp_path / "controls")
    config = exclusion_config(plan, references, groups, controls, tmp_path / "excluded")
    return references, config, identity


def test_complete_reference_precedence_exact_near_controls_and_zero_overlap(tmp_path):
    references, config, _ = fixture(tmp_path)
    result = run_exclusion(config)
    analysis = analyze_exclusion(config["output_directory"], references)
    assert analysis["exact_reference_overlap"] == 0
    assert analysis["reference_outputs_preserved"] is True
    assert analysis["controls"]["firewall_near_control"]["reason"] == "near_duplicate"
    assert analysis["controls"]["firewall_exact_control"]["reason"] == "exact_duplicate"
    assert all(value["records"] == 1 for value in analysis["retained"].values())
    assert all(
        counts["reference:exact_duplicate"] == 1
        for counts in analysis["candidate_removals"].values()
    )
    assert result["result"]["manifest"]["counts"]["records_retained"] == 18


def test_full_reference_boundary_resume_preserves_outputs(tmp_path):
    references, config, _ = fixture(tmp_path)
    with pytest.raises(RuntimeError, match="injected"):
        run_exclusion(config, crash_after_records=references["records"] + 1)
    resumed = run_exclusion(config)["result"]["manifest"]
    clean_config = {**config, "output_directory": str(tmp_path / "clean")}
    clean = run_exclusion(clean_config)["result"]["manifest"]
    for key in ("outputs", "removals", "counts"):
        assert resumed[key] == clean[key]


@pytest.mark.parametrize("change", ["order", "reference_output", "policy"])
def test_bank_parent_cannot_fake_reference_pass(tmp_path, change):
    references, config, _ = fixture(tmp_path)
    parent = copy.deepcopy(run_exclusion(config)["result"]["manifest"])
    if change == "order":
        parent["sources"][0], parent["sources"][1] = parent["sources"][1], parent["sources"][0]
    elif change == "reference_output":
        parent["outputs"][references["sources"][0]["id"]]["sha256"] = "0" * 64
    else:
        parent["policy"]["verified_jaccard_threshold"] = 0.9
    with pytest.raises(ValueError, match="reference"):
        verify_excluded_parent(parent, references)


def test_bank_v2_accepts_only_bound_excluded_category_groups(tmp_path, monkeypatch):
    references, config, identity = fixture(tmp_path)
    run_exclusion(config)
    parent = tmp_path / "excluded/manifest.json"
    tokenizer = tmp_path / "tokenizer.model"
    tokenizer.write_text("1")
    path = tmp_path / "bank-plan.json"
    plan = {
        "format": "speck_bounded_source_bank_plan",
        "format_version": 2,
        "purpose": "engineering_rehearsal_not_training_data",
        "firewall_plan": identity,
        "parent_manifest": {"path": str(parent), "sha256": file_sha256(parent)},
        "reference_tokenizer": {"path": str(tokenizer), "sha256": file_sha256(tokenizer)},
        "sources": [
            {
                "category": category,
                "parent_source_id": f"acquired_train__{category}",
                "target_utf8_bytes": 1,
            }
            for category in CATEGORIES
        ],
        "checkpoint_records": 16,
        "shard_tokens": 4096,
        "output_directory": str(tmp_path / "bank"),
    }
    write(path, plan)
    monkeypatch.setattr("speck.data.source_bank.Tokenizer", ByteTokenizer)
    result = prepare_source_bank(path)
    assert len(result["sources"]) == 6
    assert not result["training_authority"]
    plan["sources"][0]["parent_source_id"] = "acquired_train__code"
    write(path, plan)
    with pytest.raises(ValueError, match="category"):
        load_bank_plan(path)


def test_grouping_keeps_unit_order_metadata_and_rejects_input_changes(tmp_path, monkeypatch):
    from speck.data.acquisition_units import acquire_unit

    plan, original = acquisition_fixture(tmp_path, monkeypatch)
    plan["units"] = [
        {**original, "id": "first", "start_row": 0, "stop_row": 3},
        {**original, "id": "second", "start_row": 3, "stop_row": 6},
    ]
    acquired = tmp_path / "acquired"
    for unit in plan["units"]:
        acquire_unit(plan, unit, acquired, {})
    grouped = group_acquisition_units(plan, acquired, tmp_path / "groups")
    records = [
        json.loads(line) for line in (tmp_path / "groups/web.jsonl").read_text().splitlines()
    ]
    assert [record["source_row"] for record in records] == [0, 2, 4, 5]
    assert records[1]["metadata"] == {"id": "2"}
    assert group_acquisition_units(plan, acquired, tmp_path / "groups") == grouped
    (acquired / "second/records.jsonl").write_text("corruption")
    with pytest.raises(ValueError, match="identity"):
        group_acquisition_units(plan, acquired, tmp_path / "groups")
