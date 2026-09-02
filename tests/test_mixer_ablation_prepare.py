from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.mixer_ablation_prepare import (
    VARIANTS,
    ablation_summary,
    prepare,
    scale_data_config,
    scale_train_config,
    variant_architecture,
)
from speck.architecture import (
    ArchitectureConfig,
    AttentionSpec,
    GatedCausalConvSpec,
    GatedDeltaNetSpec,
)
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
    assert sum(isinstance(item, AttentionSpec) for item in mixers(variants["full-local"])) == 20
    assert all(
        item.scope == "sliding" and item.window_size == 2_048
        for item in mixers(variants["full-local"])
    )
    assert sum(isinstance(item, GatedDeltaNetSpec) for item in mixers(variants["pure-gdn"])) == 20
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


def test_prepare_materializes_inherited_configs_for_nested_candidates(tmp_path):
    output = tmp_path / "nested" / "MixerSweep"
    contract = prepare(
        SimpleNamespace(
            source_experiment=experiment,
            output_dir=output,
            window_size=2_048,
            train_tokens=None,
            data_tokens=None,
            data_experiment=None,
        )
    )

    assert set(contract["variants"]) == set(VARIANTS)
    source = load_experiment(experiment, "data", "long_context", "tokenizer", "train")
    for name in VARIANTS:
        candidate = load_experiment(
            output / name,
            "data",
            "long_context",
            "model",
            "tokenizer",
            "train",
        )
        assert candidate["data"] == source["data"]
        assert candidate["long_context"] == source["long_context"]
        assert candidate["tokenizer"] == source["tokenizer"]
        assert candidate["train"]["run"] == f"MixerSweep-{name}"
        assert "extends" not in (output / name / "data.json").read_text()
        with torch.device("meta"):
            model = SpeckForCausalLM(ArchitectureConfig.from_dict(candidate["model"]))
        assert model.parameter_count() == contract["variants"][name]["parameters"]


def test_pilot_configs_scale_horizon_cadence_and_data_phases():
    source = load_experiment(experiment, "data", "train")
    train = scale_train_config(source["train"], 500_000_000)
    data = scale_data_config(source["data"], 500_000_000)

    assert train["train_tokens"] == 500_000_000
    assert train["warmup_steps"] == 51
    assert train["eval_every"] == 488
    assert train["save_every"] == 3_815
    assert [phase["end_tokens"] for phase in data["mixture"]["phases"]] == [
        350_000_000,
        450_000_000,
        500_000_000,
    ]
    assert data["output_name"] == "SpeckLC-150M-GDN-Pilot-500000000"


def test_prepare_materializes_a_shared_pilot_corpus_and_horizon(tmp_path):
    output = tmp_path / "PilotSweep"
    contract = prepare(
        SimpleNamespace(
            source_experiment=experiment,
            output_dir=output,
            window_size=4_096,
            train_tokens=32_000_000,
            data_tokens=500_000_000,
            data_experiment=None,
        )
    )
    assert contract["train_tokens"] == 32_000_000
    assert contract["data_tokens"] == 500_000_000
    data_names = set()
    for name in VARIANTS:
        configs = load_experiment(output / name, "data", "train")
        assert configs["train"]["train_tokens"] == 32_000_000
        data_names.add(configs["data"]["output_name"])
    assert data_names == {"SpeckLC-150M-GDN-Pilot-500000000"}


def test_prepare_can_reuse_an_existing_data_experiment(tmp_path):
    output = tmp_path / "ReusedDataSweep"
    data_experiment = experiment.parent / "Speck1.5-140M"
    contract = prepare(
        SimpleNamespace(
            source_experiment=experiment,
            output_dir=output,
            window_size=4_096,
            train_tokens=500_000_000,
            data_tokens=None,
            data_experiment=data_experiment,
        )
    )

    expected = load_experiment(data_experiment, "data")["data"]
    assert contract["data_experiment"] == str(data_experiment.resolve())
    assert contract["data_tokens"] == 5_000_000_000
    for name in VARIANTS:
        assert load_experiment(output / name, "data")["data"] == expected
