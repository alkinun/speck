from pathlib import Path

import pytest
import torch

from speck.budget import estimate_context_budget
from speck.config import load_experiment
from speck.model import build_model

experiment = Path(__file__).parents[1] / "experiments" / "SpeckLC-1.2B"


def target_model():
    config = load_experiment(experiment, "model")["model"]
    with torch.device("meta"):
        return build_model(config, vocab_size=32_000)


def test_context_budget_exposes_quadratic_global_attention_cost():
    report = estimate_context_budget(
        target_model(),
        (4_096, 1_048_576),
        effective_tflops=400,
        h100_hours=10_000,
        weight_bits=4,
        kv_cache_dtype="int8",
    )
    short, long = report["points"]
    assert report["parameters"] == 1_218_451_776
    assert short["tokens_in_budget"] > 1e12
    assert long["compute_multiple_vs_shortest"] > 10
    assert long["state_by_kind"]["attention_kv"] < 3.1 * 1024**3
    assert long["weights_plus_state_bytes"] < 4 * 1024**3


def test_context_budget_rejects_cherry_picked_lengths_and_compute():
    model = target_model()
    with pytest.raises(ValueError, match="sorted and unique"):
        estimate_context_budget(
            model,
            (128, 64),
            effective_tflops=400,
            h100_hours=1,
        )
    with pytest.raises(ValueError, match="TFLOP"):
        estimate_context_budget(
            model,
            (64,),
            effective_tflops=0,
            h100_hours=1,
        )
