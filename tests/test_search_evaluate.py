import torch

from speck.model import Config, LayerConfig
from speck.search.evaluate import (
    InferenceSettings,
    QuantizationSettings,
    evaluate_inference,
    objective_values,
    quantized_weight_bytes,
)


def config():
    return Config(
        vocab_size=16,
        layers=(
            LayerConfig(8, 16, 1),
            LayerConfig(8, 16, None),
        ),
        head_dim=4,
        max_position_embeddings=10,
    )


def test_inference_records_every_context():
    metrics = evaluate_inference(
        config(),
        InferenceSettings(contexts=(4, 8), warmup_samples=0, samples=2),
        torch.device("cpu"),
        seed=7,
    )
    assert set(metrics["contexts"]) == {"4", "8"}
    assert metrics["kv_cache_bytes_per_token"] == 16
    assert len(metrics["contexts"]["4"]["prefill"]["samples_ms"]) == 2
    assert len(metrics["contexts"]["8"]["decode"]["samples_ms"]) == 2
    assert metrics["contexts"]["8"]["cache_allocated_bytes"] == 288


def test_quantized_bytes_count_tied_weight_once():
    metrics = quantized_weight_bytes(
        config(), QuantizationSettings(bits=4, group_size=4)
    )
    names = {item["name"] for item in metrics["breakdown"]}
    assert "model.embed_tokens.weight" in names
    assert "lm_head.weight" not in names
    assert metrics["total_bytes"] > 0
    assert metrics["total_bytes"] < metrics["parameters"] * 2


def test_objective_values_keep_contexts_separate():
    inference = evaluate_inference(
        config(),
        InferenceSettings(contexts=(4,), warmup_samples=0, samples=1),
        torch.device("cpu"),
        seed=7,
    )
    quantization = quantized_weight_bytes(config(), QuantizationSettings())
    objectives = objective_values(
        {"validation_nll": 2.5}, inference, quantization
    )
    assert objectives["quality.validation_nll"] == 2.5
    assert "prefill.ms.context_4" in objectives
    assert "decode.ms_per_token.context_4" in objectives
    assert objectives["memory.inference_peak_bytes.context_4"] is None
