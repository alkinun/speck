import copy
from pathlib import Path

import pytest

from speck.data import ordered_stock_exclusion as bridge
from speck.data import swh_cache
from speck.data.acquisition_units import _digest
from speck.data.firewall_integration import group_acquisition_units
from speck.data.ordered_stock_recovery import identity
from speck.data.source_stock import reuse_completed_units
from speck.data.stack_edu_ordered_units import acquire_batch
from speck.provenance.io import durable_json
from tests.data import test_stack_edu_ordered as fixtures


@pytest.fixture
def completed(tmp_path, monkeypatch):
    plan, template, targets, store = fixtures.setup.__wrapped__(tmp_path, monkeypatch)
    old, new = tmp_path / "old", tmp_path / "new"
    metadata = tmp_path / "metadata.json"
    durable_json(metadata, {})
    template.update(index_identity=identity(metadata), metadata_path=str(metadata))
    parent = tmp_path / "parent.json"
    durable_json(parent, {})
    predecessor = tmp_path / "previous.json"
    durable_json(predecessor, {"archive_directory": str(tmp_path / "old-archive")})
    plan.update(
        working_directory=str(new),
        archive_directory=str(tmp_path / "archive"),
        reuse_owner_directory=str(old),
        supersedes=identity(predecessor),
        fallback_caches=[],
        inputs={"old_stock_plan": identity(parent)},
        source_use={"identity": identity(parent)},
        language_order=["Rust"],
        targets={"Rust": {"preparation_target_tokens": 1}},
        files=[{"unit": template, "index_identity": identity(metadata)}],
    )
    items = []
    for i, (base, batch) in enumerate(((old, targets[:3]), (new, targets[3:]))):
        unit = {
            **template,
            "id": f"Rust-{i}",
            "start_row": batch[0]["source_row"],
            "stop_row": batch[-1]["source_row"] + 1,
            "targets_sha256": _digest(batch),
        }
        manifest = acquire_batch(
            plan, unit, batch, base / "acquired", store, {}, fixtures.Tokenizer()
        )
        items.append(
            {
                "unit": unit,
                "manifest": manifest,
                "manifest_identity": identity(base / "acquired" / unit["id"] / "manifest.json"),
            }
        )
    documents = sum(r["manifest"]["retained_records"] for r in items)
    tokens = sum(r["manifest"]["tokens_before_full_exclusion"] for r in items)
    acquire_plan = tmp_path / "acquisition.json"
    durable_json(acquire_plan, {})
    result = {
        "format": "speck_stack_edu_ordered_acquisition_result",
        "format_version": 2,
        "status": "content_acquisition_complete",
        "training_authority": False,
        "full_exclusion_performed": False,
        "inputs": plan["inputs"],
        "source_use": plan["source_use"]["identity"],
        "reference_tokenizer": plan["reference_tokenizer"],
        "plan": identity(acquire_plan),
        "acquired_directories": [str(p / "acquired") for p in (old, new)],
        "units": items,
        "progress": {
            "complete_units": 2,
            "state": "content_acquisition_complete",
            "unprocessed_languages": [],
            "shortfall_languages": [],
            "retained_records": documents,
            "tokens_before_full_exclusion": tokens,
            "by_language_before_full_exclusion": {
                "Rust": {"documents": documents, "tokens": tokens}
            },
        },
    }
    monkeypatch.setattr(
        swh_cache, "_get", lambda *a, **k: pytest.fail("import fetched content again")
    )
    return plan, result


def test_both_original_locations_copy_into_existing_grouping_without_fetch(completed, tmp_path):
    plan, result = completed
    units, identities = bridge.validate_ordered_units(result, plan)
    plan = {
        **plan,
        "units": units,
        "reuse_acquisition_units": identities,
        "ordered_result": "fixture",
    }
    originals = {
        p: p.read_bytes()
        for key in ("working_directory", "reuse_owner_directory")
        for p in (Path(plan[key]) / "acquired").rglob("*")
        if p.is_file()
    }
    output = tmp_path / "import"
    assert len(reuse_completed_units(plan, output, tmp_path)) == 2
    receipt = bridge.imported_units(plan, output)
    assert len(receipt["units"]) == 2
    groups = group_acquisition_units(plan, output, tmp_path / "groups")
    expected = b"".join(
        (
            Path(item["manifest_identity"]["path"]).parent / item["manifest"]["output"]["path"]
        ).read_bytes()
        for item in result["units"]
    )
    assert (tmp_path / "groups" / groups["outputs"]["code"]["path"]).read_bytes() == expected
    assert all(p.read_bytes() == data for p, data in originals.items())
    # The deliberately limited import cannot masquerade as a full fetch archive.
    assert not list(output.rglob("fetches.jsonl"))


@pytest.mark.parametrize(
    "change",
    ["partial", "source_use", "total", "duplicate", "payload", "policy", "outside", "order"],
)
def test_incomplete_changed_or_misbound_acquisition_rejected(completed, change):
    plan, result = copy.deepcopy(completed)
    if change == "partial":
        result["status"] = "pre_exclusion_language_shortfall"
    elif change == "source_use":
        result["source_use"] = {}
    elif change == "total":
        result["progress"]["retained_records"] += 1
    elif change == "duplicate":
        result["units"].append(result["units"][0])
    elif change == "payload":
        item = result["units"][0]
        (
            Path(item["manifest_identity"]["path"]).parent / item["manifest"]["output"]["path"]
        ).write_text("changed")
    elif change == "policy":
        plan["base"]["filtering"]["max_chars"] += 1
    elif change == "order":
        result["units"].reverse()
    else:
        result["units"][0]["manifest_identity"]["path"] = "/outside/manifest.json"
    with pytest.raises(ValueError):
        bridge.validate_ordered_units(result, plan)


def test_loader_inherits_exclusion_policy_and_finite_language_targets(
    completed, tmp_path, monkeypatch
):
    acquisition, result = completed
    parent = {
        "reference_parent": acquisition["inputs"]["old_stock_plan"],
        "sqlite_policy": acquisition["inputs"]["old_stock_plan"],
        "maximum_observed_wal_bytes": 2147483648,
        "target_reference_tokens": 1440000000,
    }
    monkeypatch.setattr(bridge, "load_ordered_stock", lambda p: acquisition)
    monkeypatch.setattr(bridge, "load_stack_edu_preparation", lambda p: parent)
    result_path = tmp_path / "result.json"
    durable_json(result_path, result)
    spec = {
        "format": "speck_ordered_stack_edu_exclusion_preparation",
        "format_version": 1,
        "source_id": "stack_edu",
        "ordered_result": identity(result_path),
        "output_directory": str(tmp_path / "exclusion"),
        "training_authority": False,
    }
    path = tmp_path / "exclusion-plan.json"
    durable_json(path, spec)
    compiled = bridge.load_ordered_exclusion(path)
    assert compiled["target_reference_tokens"] == 1
    assert compiled["source_language_targets"] == {"Rust": 1}
    assert compiled["maximum_observed_wal_bytes"] == parent["maximum_observed_wal_bytes"]
    assert compiled["base"] == acquisition["base"]
    assert len(compiled["reuse_acquisition_units"]) == 2
    spec["output_directory"] = acquisition["working_directory"]
    durable_json(path, spec)
    with pytest.raises(ValueError, match="overlaps"):
        bridge.load_ordered_exclusion(path)
