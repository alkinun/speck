import json
from pathlib import Path

import torch

from speck.architecture import AttentionSpec, KimiDeltaAttentionSpec, SwiGLUSpec
from speck.config import load_experiment
from speck.model import build_model

ROOT = Path(__file__).parents[1]
FLAGSHIP = ROOT / "research" / "flagship"
SHAPE_A = FLAGSHIP / "targets" / "shape-a"


def test_execution_plan_balances_the_grant_and_keeps_reserve_conditional():
    plan = json.loads((FLAGSHIP / "plan.json").read_text())
    phases = plan["phases"]

    assert plan["format"] == "speck_flagship_execution_plan"
    assert plan["status"] == "pregrant"
    assert sum(phase["gpu_hours"] for phase in phases) == 5_000
    assert sum(phase["gpu_hours"] for phase in phases if not phase.get("conditional")) == 3_985
    reserve = next(phase for phase in phases if phase["id"] == "P7")
    assert reserve["gpu_hours"] == plan["budget"]["reserve_gpu_hours"] == 1_015
    assert reserve["conditional"] is True
    assert "mixture of experts" in plan["scope"]["excluded"]
    assert "dense-width flagship" in plan["flexibility"]["binding"]


def test_data_plan_has_a_complete_english_mixture_and_matches_execution_budget():
    data_plan = json.loads((FLAGSHIP / "data_plan.json").read_text())
    execution_plan = json.loads((FLAGSHIP / "plan.json").read_text())
    categories = data_plan["categories"]
    experiments = data_plan["experiments"]

    assert data_plan["format"] == "speck_flagship_data_plan"
    assert data_plan["language_scope"]["natural_language"] == ["en"]
    assert [category["id"] for category in categories] == [
        "web",
        "code",
        "math",
        "synthetic",
        "science",
        "reference",
    ]
    assert sum(category["prior_percent"] for category in categories) == 100
    assert all(
        category["min_percent"] <= category["prior_percent"] <= category["max_percent"]
        for category in categories
    )
    assert sum(experiment["new_runs"] for experiment in experiments) == 80
    assert sum(experiment["gpu_hours"] for experiment in experiments) == 593
    assert data_plan["totals"] == {"new_runs": 80, "gpu_hours": 593}

    phase_hours = {phase["id"]: phase["gpu_hours"] for phase in execution_plan["phases"]}
    assert phase_hours["P1"] == 230
    assert phase_hours["P2"] == 440
    assert execution_plan["budget"]["mandatory_gpu_hours"] == 3_985
    assert execution_plan["budget"]["reserve_gpu_hours"] == 1_015


def test_data_plan_preserves_selection_firewall_and_replication():
    data_plan = json.loads((FLAGSHIP / "data_plan.json").read_text())
    experiments = {experiment["id"]: experiment for experiment in data_plan["experiments"]}
    selection = data_plan["selection"]

    assert data_plan["partitions"] == [
        "tokenizer_sample",
        "selection_heldout",
        "sealed_audit",
    ]
    assert selection["primary_unit"] == "bits_per_utf8_byte"
    assert selection["e2a_advance"] == experiments["E2b"]["new_runs"] == 6
    assert selection["e2b_advance"] * selection["final_confirmation_seeds"] == experiments[
        "E2c"
    ]["new_runs"]
    assert experiments["E5"]["reused_control_runs"] == 2


def test_execution_plan_dependencies_are_acyclic_and_days_fit_the_grant():
    plan = json.loads((FLAGSHIP / "plan.json").read_text())
    positions = {phase["id"]: index for index, phase in enumerate(plan["phases"])}

    assert len(positions) == len(plan["phases"])
    for phase in plan["phases"]:
        assert 0 <= phase["days"][0] <= phase["days"][1] <= plan["calendar_days"]
        assert all(positions[parent] < positions[phase["id"]] for parent in phase["depends_on"])


def test_shape_a_is_current_bounded_and_deliberately_not_launchable():
    configs = load_experiment(SHAPE_A, "long_context", "model")
    target = json.loads((SHAPE_A / "target.json").read_text())
    with torch.device("meta"):
        model = build_model(configs["model"], vocab_size=32_000)

    mixers = [
        stage.branches[0]
        for invocation in model.execution_plan
        for stage in invocation.block.stages[:1]
    ]
    feed_forwards = [
        branch
        for invocation in model.execution_plan
        for stage in invocation.block.stages
        for branch in stage.branches
        if isinstance(branch, SwiGLUSpec)
    ]
    attention = [branch for branch in mixers if isinstance(branch, AttentionSpec)]

    assert model.parameter_count() == target["parameters"] == 1_195_878_432
    assert sum(isinstance(branch, KimiDeltaAttentionSpec) for branch in mixers) == 18
    assert len(attention) == 6
    assert all((branch.num_key_value_heads, branch.rope_dim) == (4, 0) for branch in attention)
    assert len(feed_forwards) == 24
    assert all(branch.intermediate_size == 5_120 for branch in feed_forwards)
    assert model.config.max_position_embeddings == 131_072
    assert configs["long_context"]["lengths"][-1] == 131_072
    assert target["status"] == "planning_default_not_launchable"
    assert not (SHAPE_A / "data.json").exists()
    assert not (SHAPE_A / "train.json").exists()


def test_storage_readiness_result_clears_the_pregrant_threshold():
    result = json.loads(
        (ROOT / "results" / "storage" / "checkpoint-relocation-20260906.json").read_text()
    )

    assert result["format"] == "speck_checkpoint_storage_relocation"
    assert result["relocated"]["bytes"] > 80_000_000_000
    assert result["verification"]["checksum_dry_run_changes_per_family"] == 0
    assert result["verification"]["latest_checkpoint_metadata_loaded_through_symlink"] is True
    assert result["after"]["root_use_percent"] < 80
