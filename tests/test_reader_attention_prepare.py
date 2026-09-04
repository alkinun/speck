from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.reader_attention_prepare import (
    attention_slots,
    memory_plan,
    prepare,
    shared_memory_architecture,
)
from speck.architecture import ArchitectureConfig, AttentionSpec
from speck.config import load_experiment
from speck.model import SpeckForCausalLM

repository = Path(__file__).parents[1]
source_experiment = (
    repository / "experiments" / "SpeckLC-150M-KimiTransfer131M" / "kda-sigmoid-nope"
)


def source_architecture():
    return ArchitectureConfig.from_dict(load_experiment(source_experiment, "model")["model"])


def attention_branches(config):
    return [
        branch
        for invocation in config.execution_plan
        for stage in invocation.block.stages
        for branch in stage.branches
        if isinstance(branch, AttentionSpec)
    ]


def test_memory_plan_places_writers_before_the_readers_they_serve():
    assert memory_plan(5, 5) == [("write", f"global_{index}") for index in range(5)]
    assert memory_plan(5, 1) == [("write", "global_0")] + [("read", "global_0")] * 4
    assert memory_plan(5, 2) == [
        ("write", "global_0"),
        ("read", "global_0"),
        ("write", "global_1"),
        ("read", "global_1"),
        ("read", "global_1"),
    ]


def test_memory_plan_rejects_more_caches_than_attention_slots():
    with pytest.raises(ValueError, match="cannot exceed"):
        memory_plan(5, 6)
    with pytest.raises(ValueError, match="cache count"):
        memory_plan(5, 0)


def test_source_architecture_exposes_five_global_attention_slots():
    assert len(attention_slots(source_architecture())) == 5


@pytest.mark.parametrize("caches", (5, 3, 2, 1))
def test_cache_arms_match_source_flops_and_matrix_parameters(caches):
    source = source_architecture()
    with torch.device("meta"):
        baseline = SpeckForCausalLM(source)
    config, plan = shared_memory_architecture(source, caches)
    with torch.device("meta"):
        model = SpeckForCausalLM(config)
    head_dim = attention_branches(source)[0].head_dim
    assert plan["readers"] == 5 - caches
    assert plan["unmatched_parameters_in_feed_forward"] == 0
    assert model.flops_per_token(4_096) == baseline.flops_per_token(4_096)
    assert baseline.parameter_count() - model.parameter_count() == plan["readers"] * head_dim


@pytest.mark.parametrize("caches", (3, 2, 1))
def test_cache_arms_allocate_exactly_one_cache_for_each_memory(caches):
    source = source_architecture()
    config, _ = shared_memory_architecture(source, caches)
    with torch.device("meta"):
        model = SpeckForCausalLM(config)
        state = model.state(length=4_096, device="meta", kv_cache_dtype=torch.bfloat16)
    branches = attention_branches(config)
    assert sum(branch.writes_memory for branch in branches) == caches
    assert sum(branch.reads_memory for branch in branches) == 5 - caches
    assert len({branch.memory for branch in branches}) == caches
    assert sum(1 for entry in state.entries.values() if hasattr(entry, "keys")) == caches


def test_five_cache_arm_reuses_the_source_architecture_exactly():
    source = source_architecture()
    config, plan = shared_memory_architecture(source, 5)
    assert plan["reuses_source_architecture"]
    assert config.settings() == source.settings()


def test_multi_query_writer_reports_its_unmatched_accounting():
    source = source_architecture()
    config, plan = shared_memory_architecture(source, 1, writer_key_value_heads=1)
    assert plan["unmatched_parameters_in_feed_forward"] > 0
    assert {branch.num_key_value_heads for branch in attention_branches(config)} == {1}


def test_prepare_materializes_a_verified_cache_staircase(tmp_path):
    output = tmp_path / "SpeckLC-150M-ReaderAttention131M"
    contract = prepare(
        SimpleNamespace(
            source_experiment=source_experiment,
            output_dir=output,
            caches=[5, 2, 1],
            mqa_caches=[],
            train_tokens=32_000_000,
            seed=43,
        )
    )
    assert contract["source_resident_state_bytes"]["131072"]["total_bytes"] == 504_860_160
    assert set(contract["variants"]) == {"caches-5", "caches-2", "caches-1"}
    single = contract["variants"]["caches-1"]
    assert single["resident_state_bytes"]["131072"]["total_bytes"] == 102_206_976
    assert single["resident_state_bytes"]["131072"]["by_kind"]["attention_kv"] == 100_663_296
    assert single["matrix_parameter_and_flop_matched"]
    assert single["flops_delta_versus_source"] == 0
    for name in contract["variants"]:
        candidate = load_experiment(output / name, "data", "model", "tokenizer", "train")
        assert candidate["train"]["train_tokens"] == 32_000_000
        assert candidate["train"]["seed"] == 43
        assert candidate["train"]["run"] == f"SpeckLC-150M-ReaderAttention131M-{name}"
        assert candidate["train"]["wandb_group"] == "SpeckLC-150M-ReaderAttention131M"
        with torch.device("meta"):
            model = SpeckForCausalLM(ArchitectureConfig.from_dict(candidate["model"]))
        assert model.parameter_count() == contract["variants"][name]["parameters"]


def test_prepare_refuses_to_overwrite_an_existing_staircase(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        prepare(
            SimpleNamespace(
                source_experiment=source_experiment,
                output_dir=output,
                caches=[1],
                mqa_caches=[],
                train_tokens=None,
                seed=None,
            )
        )
