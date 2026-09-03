from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.kimi_transfer_prepare import (
    VARIANTS,
    prepare,
    staircase_summary,
    variant_architecture,
)
from speck.architecture import (
    ArchitectureConfig,
    AttentionSpec,
    GatedDeltaNetSpec,
    KimiDeltaAttentionSpec,
)
from speck.config import load_experiment
from speck.model import SpeckForCausalLM

repository = Path(__file__).parents[1]
source_experiment = (
    repository / "experiments" / "SpeckLC-150M-MixerScreen-131M" / "gdn-global"
)


def mixers(config):
    return [invocation.block.stages[0].branches[0] for invocation in config.execution_plan]


def source_architecture():
    return ArchitectureConfig.from_dict(load_experiment(source_experiment, "model")["model"])


def test_kimi_transfer_staircase_is_one_intervention_at_each_step():
    source = source_architecture()
    variants = {name: variant_architecture(source, name) for name in VARIANTS}

    for config in variants.values():
        assert config.logical_depth == source.logical_depth == 20
        assert config.expected_parameters is not None
        assert sum(isinstance(mixer, AttentionSpec) for mixer in mixers(config)) == 5

    fla_silu = mixers(variants["gdn-fla-silu-rope"])
    assert all(
        mixer.decay_initialization == "fla" and mixer.output_gate_activation == "silu"
        for mixer in fla_silu
        if isinstance(mixer, GatedDeltaNetSpec)
    )
    fla_sigmoid = mixers(variants["gdn-fla-sigmoid-rope"])
    assert all(
        mixer.decay_initialization == "fla" and mixer.output_gate_activation == "sigmoid"
        for mixer in fla_sigmoid
        if isinstance(mixer, GatedDeltaNetSpec)
    )
    nope = mixers(variants["gdn-fla-sigmoid-nope"])
    assert all(
        mixer.rope_dim == 0 for mixer in nope if isinstance(mixer, AttentionSpec)
    )
    kda = mixers(variants["kda-sigmoid-nope"])
    assert sum(isinstance(mixer, KimiDeltaAttentionSpec) for mixer in kda) == 15
    assert all(
        mixer.rope_dim == 0 for mixer in kda if isinstance(mixer, AttentionSpec)
    )


def test_kimi_transfer_summary_is_compute_accounted():
    source = source_architecture()
    variants = {name: variant_architecture(source, name) for name in VARIANTS}
    summary = staircase_summary(source, variants, 4_096, 131_072_000)
    with torch.device("meta"):
        source_model = SpeckForCausalLM(source)
    target = source_model.flops_per_token(4_096) * 131_072_000
    for name, values in summary.items():
        with torch.device("meta"):
            model = SpeckForCausalLM(variants[name])
        assert values["parameters"] == model.parameter_count()
        assert values["compute_matched_tokens"] == target // values["flops_per_token"]


def test_prepare_materializes_shared_training_contract(tmp_path):
    output = tmp_path / "KimiTransfer"
    contract = prepare(
        SimpleNamespace(source_experiment=source_experiment, output_dir=output)
    )
    assert set(contract["variants"]) == set(VARIANTS)
    source = load_experiment(source_experiment, "data", "long_context", "tokenizer", "train")
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
        assert candidate["train"]["seed"] == 42
        assert candidate["train"]["train_tokens"] == 131_072_000
        assert candidate["train"]["run"] == f"KimiTransfer-{name}"
        with torch.device("meta"):
            model = SpeckForCausalLM(ArchitectureConfig.from_dict(candidate["model"]))
        assert model.parameter_count() == contract["variants"][name]["parameters"]
