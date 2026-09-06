import json
from pathlib import Path

import pytest

from scripts.moe_screen_prepare import CORPUS, arguments, prepare
from speck.architecture import ArchitectureConfig, RoutedSwiGLUSpec, SwiGLUSpec

SOURCE = Path("experiments/SpeckLC-150M-KimiTransfer131M/kda-sigmoid-nope")


@pytest.fixture(scope="module")
def screen(tmp_path_factory):
    output = tmp_path_factory.mktemp("screen") / "MoEScreen"
    contract = prepare(arguments([str(SOURCE), str(output), "--train-tokens", "500000000"]))
    return output, contract


def test_every_arm_is_materialized_with_a_complete_config(screen):
    output, contract = screen
    expected = {"dense", "g8", "g16", "g32", "p-dense-first", "p-interleaved", "p-shared"}
    assert set(contract["arms"]) == expected
    for name in expected:
        for filename in ("model.json", "data.json", "tokenizer.json", "train.json"):
            assert (output / name / filename).is_file()
    assert json.loads((output / "screen.json").read_text())["corpus"] == CORPUS


def test_arms_hold_active_feed_forward_parameters_fixed(screen):
    _, contract = screen
    control = contract["arms"]["dense"]["matched_active_parameters"]
    for name, arm in contract["arms"].items():
        assert arm["matched_active_parameters"] - arm["matched_active_delta"] == control
        # Only the shared-expert arm may differ, by one normalization per block.
        assert arm["matched_active_delta"] == (15_360 if name == "p-shared" else 0)


def test_granularity_arms_hold_total_expert_parameters_fixed(screen):
    _, contract = screen
    # Total differs across granularities only by router width, so the expert banks
    # themselves must be identical. Otherwise granularity is confounded with capacity.
    banks = {
        name: contract["arms"][name]["parameters"] - contract["arms"][name]["router_parameters"]
        for name in ("g8", "g16", "g32")
    }
    assert len(set(banks.values())) == 1
    assert contract["arms"]["p-shared"]["parameters"] == contract["arms"]["g8"]["parameters"]
    assert contract["arms"]["dense"]["parameters"] < min(banks.values())


def test_granularity_arms_scale_expert_width_against_expert_count(screen):
    output, contract = screen
    width = contract["dense_feed_forward_width"]
    for name, experts, selected in (("g8", 8, 2), ("g16", 16, 4), ("g32", 32, 8)):
        config = ArchitectureConfig.from_dict(
            json.loads((output / name / "model.json").read_text())
        )
        routed = [
            branch
            for group in config.blocks
            for stage in group.block.stages
            for branch in stage.branches
            if isinstance(branch, RoutedSwiGLUSpec)
        ]
        assert routed and all(
            (branch.num_experts, branch.top_k, branch.intermediate_size)
            == (experts, selected, width // selected)
            for branch in routed
        )


def test_placement_arms_route_the_intended_blocks(screen):
    output, contract = screen

    def routed_blocks(name):
        config = ArchitectureConfig.from_dict(
            json.loads((output / name / "model.json").read_text())
        )
        return {
            index
            for index, group in enumerate(config.blocks)
            for stage in group.block.stages
            for branch in stage.branches
            if isinstance(branch, RoutedSwiGLUSpec)
        }

    blocks = len(
        ArchitectureConfig.from_dict(
            json.loads((output / "dense" / "model.json").read_text())
        ).blocks
    )
    assert routed_blocks("g8") == set(range(blocks))
    assert routed_blocks("p-dense-first") == set(range(2, blocks))
    assert routed_blocks("p-interleaved") == set(range(0, blocks, 2))
    assert not routed_blocks("dense")


def test_shared_arm_pairs_one_routed_and_one_always_on_expert(screen):
    output, _ = screen
    config = ArchitectureConfig.from_dict(
        json.loads((output / "p-shared" / "model.json").read_text())
    )
    stages = [
        stage
        for group in config.blocks
        for stage in group.block.stages
        if any(isinstance(branch, RoutedSwiGLUSpec) for branch in stage.branches)
    ]
    assert stages
    for stage in stages:
        routed = next(b for b in stage.branches if isinstance(b, RoutedSwiGLUSpec))
        shared = next(b for b in stage.branches if isinstance(b, SwiGLUSpec))
        assert routed.top_k == 1
        assert routed.intermediate_size == shared.intermediate_size


def test_training_recipe_is_shared_and_routing_aware(screen):
    output, contract = screen
    recipes = {
        name: json.loads((output / name / "train.json").read_text()) for name in contract["arms"]
    }
    ignored = {"run", "device_batch_size"}
    reference = {k: v for k, v in recipes["dense"].items() if k not in ignored}
    for recipe in recipes.values():
        assert {k: v for k, v in recipe.items() if k not in ignored} == reference
    assert reference["train_tokens"] == 500_000_000
    assert reference["sequence_length"] == 2_048
    assert reference["load_balance_coefficient"] == 0.01
    assert reference["router_z_loss_coefficient"] == 0.001
    assert recipes["dense"]["device_batch_size"] == 8
    assert reference["activation_checkpointing"] is True
    assert all(recipes[name]["device_batch_size"] == 4 for name in recipes if name != "dense")


def test_preparation_refuses_to_overwrite(screen, tmp_path):
    output, _ = screen
    with pytest.raises(FileExistsError):
        prepare(arguments([str(SOURCE), str(output)]))
