import json

import pytest

from speck.config import load_experiment
from speck.dataloader import source_selection_counts
from speck.dataset import validate_data_settings


def test_load_experiment(tmp_path):
    (tmp_path / "model.json").write_text(json.dumps({"hidden_size": 16}))
    assert load_experiment(tmp_path, "model") == {"model": {"hidden_size": 16}}


def test_load_experiment_requires_objects(tmp_path):
    (tmp_path / "model.json").write_text("[]")
    with pytest.raises(ValueError, match="must be an object"):
        load_experiment(tmp_path, "model")


def test_speck1_instruct_experiment_is_separate_from_base():
    base = load_experiment("experiments/Speck1-140M", "model", "tokenizer")
    instruct = load_experiment(
        "experiments/Speck1-140M-Instruct", "model", "tokenizer", "sft"
    )

    assert instruct["model"] == base["model"]
    assert instruct["tokenizer"] == base["tokenizer"]
    assert instruct["sft"]["run"] == "Speck1-140M-Instruct"


def test_speck1_1_sft_experiment_uses_speckchat2_and_original_instruct_config():
    current = load_experiment(
        "experiments/Speck1-140M-Instruct", "model", "tokenizer", "sft"
    )
    updated = load_experiment(
        "experiments/Speck1.1-140M-Instruct", "model", "tokenizer", "sft"
    )

    assert updated["model"] == current["model"]
    assert updated["tokenizer"] == current["tokenizer"]
    assert updated["sft"]["dataset"] == {
        "expected_samples": 500_000,
        "files": [
            "data/train-00000-of-00004.parquet",
            "data/train-00001-of-00004.parquet",
            "data/train-00002-of-00004.parquet",
            "data/train-00003-of-00004.parquet",
        ],
        "repo": "specklabs/SpeckChat2",
        "revision": "7b497b3e0c7f4653278cc67af27722b20a5c8d10",
        "validation_samples": 1_000,
    }
    assert updated["sft"]["pretrained"] == current["sft"]["pretrained"]
    assert updated["sft"]["epochs"] == 1
    assert updated["sft"]["run"] == "Speck1.1-140M-Instruct"


def test_speck1_1_140m_two_epoch_variant_only_changes_training_length():
    current = load_experiment(
        "experiments/Speck1.1-140M-Instruct", "model", "tokenizer", "sft"
    )
    two_epoch = load_experiment(
        "experiments/Speck1.1-140M-Instruct-2ep", "model", "tokenizer", "sft"
    )

    assert two_epoch["model"] == current["model"]
    assert two_epoch["tokenizer"] == current["tokenizer"]
    assert two_epoch["sft"]["dataset"] == current["sft"]["dataset"]
    assert two_epoch["sft"]["pretrained"] == current["sft"]["pretrained"]
    assert two_epoch["sft"]["epochs"] == 2
    assert two_epoch["sft"]["run"] == "Speck1.1-140M-Instruct-2ep"


def test_speck1_5_uses_the_original_model_and_stationary_quality_mixture():
    original = load_experiment("experiments/Speck1-140M", "model", "tokenizer", "train")
    updated = load_experiment("experiments/Speck1.5-140M", "data", "model", "tokenizer", "train")

    assert updated["model"] == original["model"]
    assert updated["tokenizer"] == original["tokenizer"]
    assert updated["train"] == {**original["train"], "run": "Speck1.5-140M"}
    data = dict(updated["data"])
    assert data.pop("output_dir") is None
    assert data.pop("output_name") == "Speck1.5-140M"
    data.pop("seed")
    validated = validate_data_settings(**data)
    assert validated["quotas"] == {
        "dclm_edu": 1_950_000_000,
        "ultra_fineweb": 1_950_000_000,
        "stack_v3": 400_000_000,
        "math_multi_style": 250_000_000,
        "math_textbook_exercise": 250_000_000,
        "ufw_l3_multi_style": 150_000_000,
        "ufw_l3_qa": 50_000_000,
    }
    assert len(validated["phases"]) == 1
    assert validated["train_reserve_tokens_per_source"] == 131_072
    sources = {source["id"]: source for source in validated["sources"]}
    assert sources["dclm_edu"]["filters"] == {
        "language": "en",
        "min_score": 3.5,
        "score_operator": ">",
    }
    assert sources["ultra_fineweb"]["filters"] == {
        "min_score": 0.7,
        "score_operator": ">",
    }
    assert sources["stack_v3"]["content_format"] == "stack_v3_repository_v1"
    assert sources["math_multi_style"]["language_detector"] == "py3langid"
    assert sources["math_textbook_exercise"]["language_detector"] == "py3langid"
    assert {source_id: source["revision"] for source_id, source in sources.items()} == {
        "dclm_edu": "dbad8ad71224482740cd9c9d353591adbf62fe04",
        "ultra_fineweb": "02c85641e3d19a854be2e09139c25adaa9518063",
        "stack_v3": "df4b205fbba4cc1c2fd1f205b10d66f730798bb9",
        "math_multi_style": "fe10db8efd35597fd7fcff8ff576b5ec4ea5ff87",
        "math_textbook_exercise": "fe10db8efd35597fd7fcff8ff576b5ec4ea5ff87",
        "ufw_l3_multi_style": "bc3b1ba986fcaef6871b9790a413b16267c2de0f",
        "ufw_l3_qa": "bc3b1ba986fcaef6871b9790a413b16267c2de0f",
    }

    schedule = {
        "requested_train_tokens": updated["data"]["requested_train_tokens"],
        "mixture": {"phases": validated["phases"]},
        "sources": [{"id": source["id"]} for source in validated["sources"]],
    }
    batch_tokens = updated["train"]["batch_tokens"]
    consumed_tokens = (
        (updated["train"]["train_tokens"] + batch_tokens - 1) // batch_tokens * batch_tokens
    )
    for world_size in (1, 2, 4, 8):
        stride = (
            updated["train"]["device_batch_size"] * updated["train"]["sequence_length"] * world_size
        )
        counts = source_selection_counts(schedule, "train", consumed_tokens, stride)
        for source_id, count in counts.items():
            assert count * stride + 1 <= (
                validated["quotas"][source_id] + validated["train_reserve_tokens_per_source"]
            )
