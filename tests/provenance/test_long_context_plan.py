"""Check allocation consistency and preserved evidence across the scope transition."""

import hashlib
import json
import subprocess

from speck.provenance.repository import repository_root

ROOT = repository_root()


def read(path):
    return json.loads((ROOT / path).read_text())


def test_selected_plan_balances_subprograms_and_has_no_dependency_cycle():
    catalog = read("research/catalog.json")
    plan = read(catalog["active_contracts"]["execution"]["path"])
    phases = {phase["id"]: phase for phase in plan["phases"]}
    assert len(phases) == len(plan["phases"])
    assert sum(p["gpu_hours"] for p in phases.values()) == plan["budget"]["gpu_hours"]
    assert sum(p["gpu_hours"] for p in phases.values() if not p["conditional"]) == 4111
    assert phases["RESERVE"]["conditional"] and phases["RESERVE"]["gpu_hours"] == 889
    assert sum(phases[f"R{i}"]["gpu_hours"] for i in range(5)) == 750
    assert sum(phases[f"V{i}"]["gpu_hours"] for i in range(1, 4)) == 236
    resolved = set()
    while len(resolved) < len(phases):
        ready = {
            key
            for key, phase in phases.items()
            if key not in resolved and set(phase["depends_on"]) <= resolved
        }
        assert ready, "phase dependencies contain a cycle or missing phase"
        resolved.update(ready)
    for role, path in plan["active_contracts"].items():
        assert catalog["active_contracts"][role]["path"] == path
        assert (ROOT / path).is_file()
    capability = read(plan["active_contracts"]["capability"])
    assert sum(s["gpu_hours"] for s in capability["stages"]) == phases["L0"]["gpu_hours"]
    assert capability["gpu_hours_ceiling"] == phases["L0"]["gpu_hours"]
    model = read(plan["active_contracts"]["model"])["flagship"]
    assert model["pretraining_gpu_hours_ceiling"] == phases["B0"]["gpu_hours"]
    assert model["base_tokens"] == plan["base_horizon_policy"]["default_tokens"]
    assert model["stretch_tokens"] == plan["base_horizon_policy"]["stretch_tokens"]


def test_study_has_complete_paired_cells_and_costs_every_parent():
    catalog = read("research/catalog.json")["active_contracts"]
    study = read(catalog["integration"]["path"])
    plan = read(catalog["execution"]["path"])
    phases = {p["id"]: p for p in plan["phases"]}
    architecture_count = len(study["factors"]["architecture"])
    training_count = len(study["factors"]["training"])
    for role in ("core", "transfer"):
        block = study[role]
        seed_count = len(block["paired_seeds"])
        assert block["parent_runs"] == seed_count * architecture_count
        assert block["branch_runs"] == seed_count * architecture_count * training_count
        assert block["gpu_hours_ceiling"] == phases[block["phase"]]["gpu_hours"]
    assert set(study["core"]["paired_seeds"]).isdisjoint(study["transfer"]["paired_seeds"])
    assert len(study["core"]["paired_seeds"]) >= 3
    assert not study["execution_ready"] and not study["training_authority"]
    data = read(catalog["data"]["path"])
    assert sum(data["base"]["category_percent"].values()) == 100
    fractions = data["long_context"]
    assert (
        fractions["default_targeted_fraction_of_assistant_tokens"]
        + fractions["general_replay_fraction_of_assistant_tokens"]
        == 1
    )


def test_pivot_preserves_original_documents_and_json_contracts():
    transition = read("research/flagship/pivot_decision_v1.json")
    snapshot = read(transition["previous_scope_snapshot"])
    for entry in snapshot["files"]:
        original = subprocess.check_output(
            ["git", "-C", str(ROOT), "show", f"{snapshot['revision']}:{entry['original_path']}"]
        )
        assert hashlib.sha256(original).hexdigest() == entry["sha256"]
        assert (
            hashlib.sha256((ROOT / entry["preserved_path"]).read_bytes()).hexdigest()
            == entry["sha256"]
        )
    for entry in transition["previous_contracts"]:
        assert hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
    old_claims = read("research/history/2026-09-15-allocation-thesis/paper/claims.json")["claims"]
    assert {c["id"] for c in old_claims} == set(transition["claim_migration"])
    claims = read("paper/claims.json")["claims"]
    assert all(c["status"] == "planned" and not c["evidence"] for c in claims)
    assert {c["id"] for c in old_claims}.isdisjoint(c["id"] for c in claims)
