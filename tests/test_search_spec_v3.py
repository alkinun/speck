import json
from pathlib import Path

import pytest

from speck.search.spec_v3 import V3SearchSettings


def settings():
    return {
        "format_version": 3,
        "seed": 42,
        "segment_plan": {"path": "~/segments.json"},
        "quality": {
            "name": "ultrafineweb_calibration",
            "sequence_length": 8,
            "batch_tokens": 32,
            "device_batch_size": 2,
            "optimizer": "muon",
            "learning_rate": 0.001,
            "minimum_learning_rate_scale": 0.1,
            "warmup_steps": 4,
            "weight_decay": 0.1,
            "gradient_clip": 1.0,
            "checkpoint_tokens": [32, 64, 128],
        },
        "calibration": {
            "noise_architectures": 4,
            "broad_architectures": 16,
            "anchor_architectures": 8,
            "initialization_seeds": 2,
            "data_seeds": 2,
            "numerical_repeats": 1,
            "noise_tokens": 32,
            "broad_tokens": 64,
            "anchor_tokens": 128,
            "bootstrap_samples": 100,
        },
        "planner": {
            "total_cost": 1000.0,
            "cost_unit": "wall_seconds",
            "max_actions_per_event": 4,
            "posterior_samples": 100,
            "surrogate_models": 8,
        },
        "space": {
            "min_logical_depth": 2,
            "max_logical_depth": 8,
            "hidden_sizes": [8, 16],
            "intermediate_sizes": [16, 32],
            "head_dims": [4, 8],
            "kv_heads": [1, 2],
            "sliding_windows": [4, 8],
            "conv_kernel_sizes": [2, 3],
            "conv_inner_sizes": [8, 16],
            "repeat_counts": [1, 2],
        },
        "objective_sets": [
            {
                "name": "gpu_short",
                "objectives": [
                    {
                        "name": "quality.target_nll",
                        "direction": "minimize",
                        "role": "quality",
                    },
                    {
                        "name": "gpu.decode_p95",
                        "direction": "minimize",
                        "role": "efficiency",
                    },
                ],
            }
        ],
        "profiles": [
            {
                "name": "gpu_short",
                "backend": "torch_native",
                "device": "cuda",
                "dtype": "bfloat16",
                "cache_dtype": "bfloat16",
                "batch_size": 1,
                "prompt_tokens": 8,
                "generated_tokens": 4,
                "warmup_requests": 1,
                "measured_requests": 5,
                "process_repetitions": 1,
            },
            {
                "name": "cpu_short",
                "backend": "torch_native",
                "device": "cpu",
                "dtype": "float32",
                "cache_dtype": "float32",
                "batch_size": 1,
                "prompt_tokens": 8,
                "generated_tokens": 4,
                "warmup_requests": 1,
                "measured_requests": 5,
                "process_repetitions": 1,
            },
        ],
    }


def test_v3_search_settings_resolve_an_immutable_training_protocol():
    parsed = V3SearchSettings.from_dict(settings())
    protocol = parsed.quality.resolve("dataset", "tokenizer", "segments")
    assert protocol.target_tokens == 128
    assert protocol.sequence_length == 8
    assert parsed.export()["format_version"] == 3
    assert V3SearchSettings.from_dict(parsed.export()) == parsed
    assert parsed.segment_plan.path.endswith("segments.json")


def test_v3_search_settings_require_both_native_hardware_views():
    value = settings()
    value["profiles"] = value["profiles"][:1]
    with pytest.raises(ValueError, match="gpu and cpu"):
        V3SearchSettings.from_dict(value)


def test_v3_search_settings_reject_invalid_calibration_panels():
    value = settings()
    value["calibration"]["anchor_architectures"] = 17
    with pytest.raises(ValueError, match="anchor"):
        V3SearchSettings.from_dict(value)
    value = settings()
    value["calibration"].update(
        initialization_seeds=1,
        data_seeds=1,
        numerical_repeats=1,
    )
    with pytest.raises(ValueError, match="seed combinations"):
        V3SearchSettings.from_dict(value)


def test_checked_v3_example_is_valid():
    path = (
        Path(__file__).parents[1]
        / "experiments"
        / "speck00-200m"
        / "search-v3.example.json"
    )
    parsed = V3SearchSettings.from_dict(json.loads(path.read_text()))
    assert parsed.calibration.broad_architectures == 32
    assert parsed.quality.checkpoint_tokens[-1] == 1_006_632_960
