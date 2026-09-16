import copy
import json
from pathlib import Path

import pytest

from scripts import acquire_stack_edu_e1s as driver
from speck.data.ordered_stock_recovery import (
    identity,
    inventory_completed,
    load_recovery_successor,
)
from speck.data.stock_blob_store import StockBlobStore
from speck.provenance.io import durable_json
from tests.data import test_stack_edu_ordered as fixtures


@pytest.fixture
def stopped(tmp_path, monkeypatch):
    plan, unit, targets, store = fixtures.setup.__wrapped__(tmp_path, monkeypatch)
    for target in targets:
        store.fetch(target)
    index = tmp_path / "index.jsonl"
    index.write_text("".join(json.dumps(t) + "\n" for t in targets))
    entry = {
        "unit": unit,
        "index_identity": identity(index),
        "index": {"eligible_rows": 5, "output": {"path": str(index)}},
    }
    output, archive = tmp_path / "work", tmp_path / "archive"
    plan.update(
        language_order=["Rust", "Go"],
        eligible_rows_per_unit=2,
        candidate_nominal_multiplier=2,
        maximum_cache_bytes=10000000,
        maximum_working_bytes=100000000,
        minimum_free_bytes=0,
        targets={
            name: {"nominal_tokens": 50, "preparation_target_tokens": 60} for name in ("Rust", "Go")
        },
        files=[entry],
    )
    spec = {
        "format": "speck_stack_edu_ordered_acquisition",
        "format_version": 1,
        "working_directory": str(output),
        "archive_directory": str(archive),
        "fallback_caches": [],
        "scope": "fixture",
        "maximum_working_bytes": 100000000,
    }
    plan_path = tmp_path / "plan.json"
    durable_json(plan_path, spec)
    driver.run_units(plan, output, archive, store, {}, fixtures.Tokenizer(), pause_after_units=1)
    (output / "owner.lock").touch()
    owner = {"plan": identity(plan_path), "repository_revision": "fixture"}
    durable_json(output / "execution.json", owner)
    durable_json(archive / "execution.json", owner)
    durable_json(output / "invocation-00000.json", {"status": "failed_preserved"})
    result = tmp_path / "inventory.json"
    inventory_completed(plan_path, result)
    successor = {
        **spec,
        "format_version": 2,
        "scope": "fixture recovery",
        "working_directory": str(tmp_path / "new"),
        "archive_directory": str(tmp_path / "new-archive"),
        "fallback_caches": [str(output / "cache")],
        "supersedes": identity(plan_path),
        "reuse_inventory": identity(result),
    }
    return plan, plan_path, successor, store


def test_recovery_preserves_prefix_and_matches_uninterrupted_final_units(stopped, tmp_path):
    plan, previous, spec, store = stopped
    output, archive = Path(spec["working_directory"]), Path(spec["archive_directory"])
    original = {
        str(p): p.read_bytes()
        for base in (tmp_path / "work", tmp_path / "archive")
        for p in base.rglob("*")
        if p.is_file()
    }
    successor = load_recovery_successor(spec, plan, identity(previous))
    new_store = StockBlobStore(
        output / "cache",
        spec["fallback_caches"],
        store.settings,
        maximum_bytes=10000000,
        minimum_free_bytes=0,
    )
    resumed = driver.run_units(successor, output, archive, new_store, {}, fixtures.Tokenizer())
    assert resumed["progress"]["reused_completed_units"] == 1
    assert len(list((output / "acquired").iterdir())) == 2
    assert all(Path(path).read_bytes() == payload for path, payload in original.items())
    clean_output = tmp_path / "clean"
    clean_output.mkdir()
    clean = driver.run_units(
        plan, clean_output, tmp_path / "clean-archive", store, {}, fixtures.Tokenizer()
    )
    assert [u["manifest"]["output"]["sha256"] for u in resumed["units"]] == [
        u["manifest"]["output"]["sha256"] for u in clean["units"]
    ]
    assert resumed["progress"]["state"] == "pre_exclusion_language_shortfall"
    assert resumed["progress"]["unprocessed_languages"] == ["Go"]
    assert new_store.used == 0


@pytest.mark.parametrize("change", ["guard", "fallback", "path"])
def test_recovery_rejects_scope_storage_and_overlap_changes(stopped, change):
    plan, previous, spec, _ = stopped
    changed = copy.deepcopy(spec)
    if change == "guard":
        changed["maximum_working_bytes"] += 1
    elif change == "fallback":
        changed["fallback_caches"] = []
    else:
        changed["working_directory"] = json.loads(previous.read_text())["working_directory"]
    with pytest.raises(ValueError):
        load_recovery_successor(changed, plan, identity(previous))


def test_recovery_rejects_changed_payload_before_any_new_unit(stopped):
    plan, previous, spec, store = stopped
    successor = load_recovery_successor(spec, plan, identity(previous))
    row = next(iter(successor["reuse_units"].values()))
    manifest = json.loads(Path(row["manifest"]["path"]).read_text())
    (Path(row["directory"]) / manifest["output"]["path"]).write_text("changed")
    output = Path(spec["working_directory"])
    output.mkdir()
    with pytest.raises(ValueError, match="payload changed"):
        driver.run_units(
            successor, output, Path(spec["archive_directory"]), store, {}, fixtures.Tokenizer()
        )
    assert not (output / "acquired").exists()


def test_inventory_requires_existing_archive_and_preserves_previous_result(stopped, tmp_path):
    _, previous, spec, _ = stopped
    with pytest.raises(FileExistsError):
        inventory_completed(previous, spec["reuse_inventory"]["path"])
    old = json.loads(previous.read_text())
    receipt = next((Path(old["archive_directory"]) / "units").glob("*/manifest.json"))
    receipt.rename(receipt.with_suffix(".saved"))
    with pytest.raises(ValueError, match="lacks its preserved archive"):
        inventory_completed(previous, tmp_path / "new-inventory.json")
    assert not receipt.exists()


def test_recovery_requires_exact_ordered_prefix(stopped):
    plan, previous, spec, store = stopped
    successor = load_recovery_successor(spec, plan, identity(previous))
    key, row = next(iter(successor["reuse_units"].items()))
    successor["reuse_units"] = {key + "_later": row}
    output = Path(spec["working_directory"])
    output.mkdir()
    with pytest.raises(ValueError, match="complete ordered prefix"):
        driver.run_units(
            successor, output, Path(spec["archive_directory"]), store, {}, fixtures.Tokenizer()
        )
    assert not (output / "acquired").exists()


def test_inventory_rejects_changed_archive_bytes(stopped, tmp_path):
    _, previous, spec, _ = stopped
    inventory = json.loads(Path(spec["reuse_inventory"]["path"]).read_text())
    receipt = json.loads(Path(inventory["units"][0]["archive_manifest"]["path"]).read_text())
    with Path(receipt["tar"]["path"]).open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="archival unit changed"):
        inventory_completed(previous, tmp_path / "changed-inventory.json")
