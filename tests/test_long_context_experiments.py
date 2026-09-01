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


def test_1_2b_target_has_six_gqa_layers_and_bounded_state():
    model, configs = load_model_on_meta("SpeckLC-1.2B")
    assert model.parameter_count() == 1_218_451_776
    assert mixer_counts(model) == (18, 6)
    state = model.state(
        length=1_048_576,
        device="meta",
        dtype=torch.bfloat16,
    )
    report = state.memory_report()
    assert report["by_kind"]["attention_kv"] == 6 * 2 * 1_048_576 * 128 * 2 * 2
    assert report["by_kind"]["gated_deltanet"] < 32 * 1024**2
    assert report["total_bytes"] < 6.1 * 1024**3
    assert configs["train"]["batch_tokens"] == 4_194_304
