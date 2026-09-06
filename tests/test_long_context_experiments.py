from pathlib import Path

import torch

from speck.architecture import AttentionSpec, GatedDeltaNetSpec
from speck.config import load_experiment
from speck.model import build_model
from speck.train import resolve_device_batch_size

root = Path(__file__).parents[1]


def load_model_on_meta(name):
    configs = load_experiment(root / "experiments" / name, "model", "tokenizer", "train")
    with torch.device("meta"):
        model = build_model(configs["model"], vocab_size=32_000)
    return model, configs


def mixer_counts(model):
    mixers = [
        branch
        for invocation in model.execution_plan
        for stage in invocation.block.stages[:1]
        for branch in stage.branches
    ]
    return (
        sum(isinstance(mixer, GatedDeltaNetSpec) for mixer in mixers),
        sum(isinstance(mixer, AttentionSpec) for mixer in mixers),
    )


def test_proxy_is_a_materialized_three_to_one_gdn_hybrid():
    model, configs = load_model_on_meta("SpeckLC-150M-GDN")
    assert model.parameter_count() == 152_916_468
    assert mixer_counts(model) == (15, 5)
    assert model.config.max_position_embeddings == 131_072
    train = configs["train"]
    for world_size in (1, 2, 4, 8):
        resolve_device_batch_size(
            train["device_batch_size"],
            train["batch_tokens"],
            train["sequence_length"],
            world_size,
        )
