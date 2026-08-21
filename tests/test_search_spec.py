import pytest

from speck.search.spec import SearchSettings, deterministic_seed


def values():
    return {
        "format_version": 2,
        "seed": 42,
        "max_architectures": 16,
        "initial_population": 4,
        "population_size": 8,
        "cohort_size": 4,
        "confidence_z": 1.645,
        "space": {
            "min_layers": 1,
            "max_layers": 3,
            "hidden_size_min": 8,
            "hidden_size_max": 12,
            "hidden_size_step": 4,
            "intermediate_size_min": 16,
            "intermediate_size_max": 24,
            "intermediate_size_step": 8,
            "kv_heads": [1, 2],
        },
        "quality": {
            "data_dir": "~/data",
            "batch_tokens": 8,
            "device_batch_size": 1,
            "eval_batch_size": 1,
            "lr": 0.001,
            "min_lr": 0.1,
            "warmup_steps": 1,
            "weight_decay": 0.1,
            "grad_clip": 1.0,
            "optimizer": "adamw",
        },
        "validation_slices": [
            {"name": "main", "offset_tokens": 0, "objective": True},
            {"name": "audit", "offset_tokens": 32, "objective": False},
        ],
        "inference": {"contexts": [4, 8], "warmup_samples": 0, "samples": 1},
        "quantization": {"bits": 4, "group_size": 4},
        "rungs": [
            {
                "name": "screen",
                "architecture_limit": 16,
                "seed_count": 1,
                "train_tokens": 16,
                "sequence_length": 4,
                "eval_every_tokens": 8,
                "eval_tokens": 8,
                "inference_samples": 1,
            },
            {
                "name": "verify",
                "architecture_limit": 4,
                "seed_count": 2,
                "train_tokens": 32,
                "sequence_length": 8,
                "eval_every_tokens": 16,
                "eval_tokens": 16,
                "inference_samples": 2,
            },
        ],
    }


def test_search_settings_resolve_rungs_and_common_seeds():
    settings = SearchSettings.from_dict(values())
    assert settings.quality.data_dir.endswith("/data")
    assert settings.quality.settings(settings.rungs[1]).train_tokens == 32
    assert settings.export()["validation_slices"][1]["name"] == "audit"
    assert deterministic_seed(42, "trial", 1, 0) == deterministic_seed(
        42, "trial", 1, 0
    )
    assert deterministic_seed(42, "trial", 1, 0) != deterministic_seed(
        42, "trial", 1, 1
    )


def test_search_settings_reject_nonincreasing_fidelity():
    settings = values()
    settings["rungs"][1]["train_tokens"] = 16
    with pytest.raises(ValueError, match="training budgets"):
        SearchSettings.from_dict(settings)
