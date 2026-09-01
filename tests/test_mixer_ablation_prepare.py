from pathlib import Path

import torch

from scripts.mixer_ablation_prepare import VARIANTS, ablation_summary, variant_architecture
from speck.architecture import ArchitectureConfig, AttentionSpec, GatedCausalConvSpec
from speck.config import load_experiment
from speck.model import SpeckForCausalLM

experiment = Path(__file__).parents[1] / "experiments" / "SpeckLC-150M-GDN"


def mixers(config):
    return [invocation.block.stages[0].branches[0] for invocation in config.execution_plan]


def test_mixer_variants_preserve_depth_and_materialize_parameter_counts():
    source = ArchitectureConfig.from_dict(load_experiment(experiment, "model")["model"])
    variants = {name: variant_architecture(source, name, 2_048) for name in VARIANTS}
    assert all(config.logical_depth == 20 for config in variants.values())
    assert all(config.expected_parameters for config in variants.values())
    assert sum(isinstance(item, AttentionSpec) for item in mixers(variants["full-global"])) == 20
    local_attention = [
        item for item in mixers(variants["gdn-local"]) if isinstance(item, AttentionSpec)
    ]
    assert len(local_attention) == 5
    assert all(item.scope == "sliding" and item.window_size == 2_048 for item in local_attention)
    assert (
        sum(isinstance(item, GatedCausalConvSpec) for item in mixers(variants["conv-global"])) == 15
    )


def test_ablation_summary_exposes_compute_matched_token_budgets():
    source = ArchitectureConfig.from_dict(load_experiment(experiment, "model")["model"])
    variants = {name: variant_architecture(source, name) for name in VARIANTS}
    summary = ablation_summary(source, variants, sequence_length=4_096, train_tokens=1_000_000)
    assert set(summary) == set(VARIANTS)
    assert all(values["compute_matched_tokens"] > 0 for values in summary.values())
    with torch.device("meta"):
        full = SpeckForCausalLM(variants["full-global"])
    assert summary["full-global"]["parameters"] == full.parameter_count()
