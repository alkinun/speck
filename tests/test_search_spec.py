import json
from pathlib import Path

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


def test_search_settings_reject_cache_dtype_mismatch():
    settings = values()
    settings["inference"]["cache_dtype_bytes"] = 1
    with pytest.raises(ValueError, match="cache dtypes"):
        SearchSettings.from_dict(settings)


def test_search_settings_reject_partial_initial_cohort():
    settings = values()
    settings["initial_population"] = 5
    with pytest.raises(ValueError, match="complete cohorts"):
        SearchSettings.from_dict(settings)


def test_search_settings_reject_partial_evaluation_batch():
    settings = values()
    settings["rungs"][1]["eval_tokens"] = 15
    with pytest.raises(ValueError, match="complete batches"):
        SearchSettings.from_dict(settings)


def test_checked_experiment_uses_multi_fidelity_search():
    experiment = Path(__file__).parents[1] / "experiments" / "speck00-200m"
    path = experiment / "search.json"
    settings = SearchSettings.from_dict(json.loads(path.read_text(encoding="utf-8")))
    assert settings.format_version == 2
    assert [rung.name for rung in settings.rungs] == ["screen", "develop", "verify"]
    assert [rung.seed_count for rung in settings.rungs] == [1, 2, 3]
    assert [rung.sequence_length for rung in settings.rungs] == [256, 512, 1024]
    assert sum(
        rung.architecture_limit * rung.seed_count for rung in settings.rungs
    ) == 216
    assert sum(
        rung.architecture_limit * rung.seed_count * rung.train_tokens
        for rung in settings.rungs
    ) == 226_492_416
    assert sum(
        rung.architecture_limit
        * rung.seed_count
        * (rung.train_tokens // rung.eval_every_tokens + 1)
        * rung.eval_tokens
        * len(settings.validation_slices)
        for rung in settings.rungs
    ) == 171_966_464
    model = json.loads((experiment / "model.json").read_text(encoding="utf-8"))
    data = json.loads((experiment / "data.json").read_text(encoding="utf-8"))
    assert settings.inference.contexts[-1] + 1 <= model["max_position_embeddings"]
    assert all(
        rung.sequence_length <= model["max_position_embeddings"]
        for rung in settings.rungs
    )
    assert all(
        item.offset_tokens + rung.eval_tokens + 1 <= data["validation_tokens"]
        for item in settings.validation_slices
        for rung in settings.rungs
    )
