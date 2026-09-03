from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.attention_gate_prepare import VARIANTS, gated_architecture, prepare
from speck.architecture import ArchitectureConfig, AttentionSpec, SwiGLUSpec
from speck.config import load_experiment
from speck.model import SpeckForCausalLM

repository = Path(__file__).parents[1]
source_experiment = (
    repository / "experiments" / "SpeckLC-150M-KimiTransfer131M" / "kda-sigmoid-nope"
)


def operations(config, kind):
    return [
        branch
        for invocation in config.execution_plan
        for stage in invocation.block.stages
        for branch in stage.branches
        if isinstance(branch, kind)
    ]


def test_gate_variants_match_parameters_and_analytic_flops_exactly():
    source = ArchitectureConfig.from_dict(load_experiment(source_experiment, "model")["model"])
    with torch.device("meta"):
        baseline = SpeckForCausalLM(source)
    expected_decrements = {"ungated": 0, "headwise": 1, "elementwise": 64}
    for name in VARIANTS:
        config, decrement = gated_architecture(source, name)
        with torch.device("meta"):
            model = SpeckForCausalLM(config)
        assert decrement == expected_decrements[name]
        assert model.parameter_count() == baseline.parameter_count()
        assert model.flops_per_token(4_096) == baseline.flops_per_token(4_096)
        expected_gate = "none" if name == "ungated" else name
        assert {branch.output_gate for branch in operations(config, AttentionSpec)} == {
            expected_gate
        }
        assert {branch.intermediate_size for branch in operations(config, SwiGLUSpec)} == {
            2_304 - decrement
        }


def test_prepare_materializes_shared_screen_contract(tmp_path):
    output = tmp_path / "AttentionGate32M"
    contract = prepare(
        SimpleNamespace(
            source_experiment=source_experiment,
            output_dir=output,
            train_tokens=32_000_000,
        )
    )
    assert set(contract["variants"]) == set(VARIANTS)
    for name in VARIANTS:
        candidate = load_experiment(output / name, "data", "model", "tokenizer", "train")
        assert candidate["train"]["train_tokens"] == 32_000_000
        assert candidate["train"]["warmup_steps"] == 3
        assert candidate["train"]["eval_every"] == 122
        assert candidate["train"]["run"] == f"AttentionGate32M-{name}"
        with torch.device("meta"):
            model = SpeckForCausalLM(ArchitectureConfig.from_dict(candidate["model"]))
        assert model.parameter_count() == contract["variants"][name]["parameters"]
