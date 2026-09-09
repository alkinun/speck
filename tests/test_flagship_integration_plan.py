import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
FLAGSHIP = ROOT / "research/flagship"


def test_active_data_successor_retires_e5_without_changing_selection_contract():
    predecessor_path = FLAGSHIP / "data_plan.json"
    predecessor = json.loads(predecessor_path.read_text())
    plan = json.loads((FLAGSHIP / "data_plan_v2.json").read_text())
    experiments = {experiment["id"]: experiment for experiment in plan["experiments"]}

    assert plan["format"] == "speck_flagship_data_plan"
    assert plan["format_version"] == 2
    assert hashlib.sha256(predecessor_path.read_bytes()).hexdigest() == plan["supersedes"]["sha256"]
    assert plan["categories"] == predecessor["categories"]
    assert plan["partitions"] == predecessor["partitions"]
    assert plan["firewall_contract"] == predecessor["firewall_contract"]
    assert plan["selection"] == predecessor["selection"]
    assert "E5" not in experiments
    assert plan["retired_experiments"][0]["id"] == "E5"
    assert plan["retired_experiments"][0]["disposition"] == "retired_before_outputs"
    assert plan["retired_experiments"][0]["former_gpu_hours"] == 100
    assert plan["totals"] == {"new_runs": 76, "gpu_hours": 493}
    assert sum(experiment["new_runs"] for experiment in experiments.values()) == 76
    assert sum(experiment["gpu_hours"] for experiment in experiments.values()) == 493
    assert plan["default_curriculum"]["claim_status"].startswith("declared operational default")


def test_integrated_validation_closes_transfer_and_assembly_gaps_without_search():
    plan = json.loads((FLAGSHIP / "integration_plan.json").read_text())
    experiments = {experiment["id"]: experiment for experiment in plan["experiments"]}

    assert plan["format"] == "speck_flagship_integrated_validation_plan"
    assert plan["gpu_hour_ceiling"] == 100
    assert set(experiments) == {"I1", "I2", "I3"}
    assert experiments["I1"]["factors"] == {
        "architecture": [
            "matched dense global-attention control",
            "C0 KDA sigmoid NoPE 3:1 hybrid",
        ],
        "data": ["balanced six-category prior", "E2c-selected stable mixture"],
    }
    assert experiments["I1"]["new_runs"] == 12
    assert experiments["I1"]["seeds"] == [42, 43, 44]
    assert experiments["I1"]["estimands"]["interaction"].startswith(
        "(selected minus prior within hybrid)"
    )
    assert "proof of no interaction" in experiments["I1"]["estimands"]["null_interpretation"]
    assert experiments["I2"]["maximum_new_runs"] == 3
    assert experiments["I2"]["seeds"] == [42, 43, 44]
    assert experiments["I2"]["analysis"]["individual_pairs_retained"] is True
    assert "complete C0" in experiments["I2"]["failure_rule"]
    assert plan["analysis_rules"]["no_reopening_E2_selection"] is True
    assert plan["analysis_rules"]["no_post_hoc_component_subset_search"] is True
    assert plan["totals"] == {"maximum_new_training_runs": 15, "gpu_hours": 100}
    assert (
        experiments["I1"]["gpu_hours"]
        + experiments["I2"]["maximum_gpu_hours"]
        + experiments["I3"]["gpu_hours"]
        == 100
    )


def test_execution_budget_reassigns_e5_hours_without_touching_reserve():
    execution = json.loads((FLAGSHIP / "plan.json").read_text())
    data = json.loads((FLAGSHIP / "data_plan_v2.json").read_text())
    integration = json.loads((FLAGSHIP / "integration_plan.json").read_text())
    architecture = json.loads((FLAGSHIP / "architecture_plan.json").read_text())
    phases = {phase["id"]: phase for phase in execution["phases"]}

    assert execution["active_contracts"] == {
        "paper": "research/flagship/PAPER.md",
        "data": "research/flagship/data_plan_v2.json",
        "architecture": "research/flagship/architecture_plan.json",
        "integration": "research/flagship/integration_plan.json",
        "tokenizer": "research/flagship/tokenizer_plan_v4.json",
    }
    assert all((ROOT / path).is_file() for path in execution["active_contracts"].values())
    assert sum(phase["gpu_hours"] for phase in phases.values()) == 5_000
    assert (
        sum(phase["gpu_hours"] for phase in phases.values() if not phase.get("conditional"))
        == 4_111
    )
    assert phases["P7"]["gpu_hours"] == execution["budget"]["reserve_gpu_hours"] == 889
    assert sum(phases[phase]["gpu_hours"] for phase in ("P1", "P2", "P3")) == (
        data["totals"]["gpu_hours"]
        + integration["totals"]["gpu_hours"]
        + architecture["totals"]["architecture_gpu_hours"]
    )
    assert (
        "integrated data-architecture and assembled-recipe validation"
        in execution["flexibility"]["forbidden_cuts"]
    )
    assert all("E5" not in cut for cut in execution["flexibility"]["cut_order"])


def test_scale_ladder_is_bound_to_integrated_assembly_confirmation():
    architecture = json.loads((FLAGSHIP / "architecture_plan.json").read_text())

    assert architecture["integrated_validation"]["scale_ladder_requires_I2"] is True
    assert architecture["integrated_validation"]["gpu_hours_outside_architecture_ceiling"] == 100
    assert all(
        stage["depends_on"] == "I2 assembled-recipe confirmation"
        for stage in architecture["scale_program"]
    )


def test_v4_tokenizer_is_a_scope_only_successor_of_the_tied_v3_contract():
    predecessor_path = FLAGSHIP / "tokenizer_plan_v3.json"
    predecessor = json.loads(predecessor_path.read_text())
    plan = json.loads((FLAGSHIP / "tokenizer_plan_v4.json").read_text())

    assert plan["format_version"] == 4
    assert hashlib.sha256(predecessor_path.read_bytes()).hexdigest() == plan["supersedes"]["sha256"]
    assert plan["candidates"] == predecessor["candidates"]
    assert plan["trainer"] == predecessor["trainer"]
    assert plan["model_accounting"] == predecessor["model_accounting"]
    assert plan["language_model_pilot"] == predecessor["language_model_pilot"]
    assert "integrated validation I1 and I2" in plan["freeze_before"]
    assert all("E1-E5" not in value for value in plan["freeze_before"])
