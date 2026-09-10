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
    predecessor_path = FLAGSHIP / "integration_plan.json"
    plan = json.loads((FLAGSHIP / "integration_plan_v2.json").read_text())
    experiments = {experiment["id"]: experiment for experiment in plan["experiments"]}

    assert plan["format"] == "speck_flagship_integrated_validation_plan"
    assert plan["format_version"] == 2
    assert hashlib.sha256(predecessor_path.read_bytes()).hexdigest() == plan["supersedes"]["sha256"]
    assert plan["gpu_hour_ceiling"] == 122
    assert set(experiments) == {"I1", "I2", "I3"}
    assert experiments["I1"]["factors"] == {
        "architecture": [
            "matched dense global-attention control",
            "C0 KDA sigmoid NoPE 3:1 hybrid",
        ],
        "data": ["balanced six-category prior", "E2c-selected stable mixture"],
    }
    assert experiments["I1"]["new_runs"] == 20
    assert experiments["I1"]["seeds"] == [42, 43, 44, 45, 46]
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
    assert plan["totals"] == {"maximum_new_training_runs": 23, "gpu_hours": 122}
    assert (
        experiments["I1"]["gpu_hours"]
        + experiments["I2"]["maximum_gpu_hours"]
        + experiments["I3"]["gpu_hours"]
        == 122
    )


def test_execution_budget_strengthens_interaction_and_preserves_total():
    execution_predecessor = FLAGSHIP / "plan.json"
    architecture_predecessor = FLAGSHIP / "architecture_plan.json"
    execution = json.loads((FLAGSHIP / "plan_v2.json").read_text())
    data = json.loads((FLAGSHIP / "data_plan_v2.json").read_text())
    integration = json.loads((FLAGSHIP / "integration_plan_v2.json").read_text())
    architecture = json.loads((FLAGSHIP / "architecture_plan_v2.json").read_text())
    phases = {phase["id"]: phase for phase in execution["phases"]}

    assert (
        hashlib.sha256(execution_predecessor.read_bytes()).hexdigest()
        == execution["supersedes"]["sha256"]
    )
    assert (
        hashlib.sha256(architecture_predecessor.read_bytes()).hexdigest()
        == architecture["supersedes"]["sha256"]
    )
    assert execution["active_contracts"] == {
        "paper": "research/flagship/PAPER.md",
        "data": "research/flagship/data_plan_v2.json",
        "architecture": "research/flagship/architecture_plan_v2.json",
        "integration": "research/flagship/integration_plan_v2.json",
        "tokenizer": "research/flagship/tokenizer_plan_v5.json",
        "paper_claims": "paper/claims.json",
        "release_policy": "research/flagship/release_and_data_use_policy_v1.json",
        "source_registry": "research/flagship/source_registry_v2.json",
        "source_rights": "research/flagship/source_rights_acceptance_v1.json",
        "data_calibration": "research/flagship/data_calibration_2b_v1/production_plan.json",
        "data_rehearsal": "research/flagship/data_rehearsal_plan_v9.json",
        "data_operations_fallback": "research/flagship/production_data_calibration_fallback_v1.json",
        "data_launch": "research/flagship/data_launch_plan_v2.json",
        "lab_direction": "research/DIRECTION.md",
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
    reconciliation = execution["flagship_budget_reconciliation"]
    assert reconciliation["ideal_gpu_hours_for_400B_at_350_tflops_per_gpu"] < 2_425
    assert reconciliation["gpu_hours_if_a_separate_1_06_multiplier_is_applied"] > 2_425
    assert reconciliation["fallback"].startswith("freeze 320B")


def test_scale_ladder_is_bound_to_integrated_assembly_confirmation():
    architecture = json.loads((FLAGSHIP / "architecture_plan_v2.json").read_text())

    assert architecture["integrated_validation"]["scale_ladder_requires_I2"] is True
    assert architecture["integrated_validation"]["gpu_hours_outside_architecture_ceiling"] == 122
    assert all(
        stage["depends_on"] == "I2 assembled-recipe confirmation"
        for stage in architecture["scale_program"]
    )
    sentinel = next(stage for stage in architecture["scale_program"] if stage["id"] == "S2")
    assert sentinel["new_runs"] == 2
    assert sentinel["gpu_hours"] == 85
    assert "not flagship-size selection" in sentinel["authority"]
    assert architecture["flagship_target"]["selected"] == "flagship-1.2b"
    assert architecture["flagship_target"]["throughput_fallback_tokens"] == 320_000_000_000


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
