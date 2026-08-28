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


def test_speck1_5_uses_the_original_model_and_phased_corpus_mixture():
    original = load_experiment("experiments/Speck1-140M", "model", "tokenizer", "train")
    updated = load_experiment("experiments/Speck1.5-140M", "data", "model", "tokenizer", "train")

    assert updated["model"] == original["model"]
    assert updated["tokenizer"] == original["tokenizer"]
    assert updated["train"] == {
        **original["train"],
        "run": "Speck1.5-140M",
        "save_every": 3815,
    }
    data = dict(updated["data"])
    assert data.pop("output_dir") is None
    assert data.pop("output_name") == "Speck1.5-140M"
    data.pop("seed")
    validated = validate_data_settings(**data)
    assert validated["quotas"] == {
        "fineweb_edu": 1_390_000_000,
        "dclm_edu": 900_000_000,
        "ultra_fineweb": 570_000_000,
        "dclm": 260_000_000,
        "finemath_4plus": 455_000_000,
        "math_textbook_exercise": 140_000_000,
        "math_multi_style": 90_000_000,
        "wikimedia": 215_000_000,
        "pes2o": 250_000_000,
        "ufw_l3_multi_style": 335_000_000,
        "cosmopedia_v2": 395_000_000,
    }
    assert len(validated["phases"]) == 3
    assert validated["train_reserve_tokens_per_source"] == 262_144
    assert [source["id"] for source in validated["sources"]] == [
        "finemath_4plus",
        "math_textbook_exercise",
        "math_multi_style",
        "cosmopedia_v2",
        "ufw_l3_multi_style",
        "pes2o",
        "wikimedia",
        "dclm_edu",
        "fineweb_edu",
        "ultra_fineweb",
        "dclm",
    ]
    sources = {source["id"]: source for source in validated["sources"]}
    assert sources["dclm_edu"]["filters"] == {
        "language": "en",
        "min_score": 3.5,
        "score_operator": ">",
    }
    assert sources["math_multi_style"]["language_detector"] == "py3langid"
    assert sources["math_textbook_exercise"]["language_detector"] == "py3langid"
    assert sources["pes2o"]["file_format"] == "jsonl_gzip"
    assert sources["pes2o"]["files"] == [
        f"data/v2/train-{index:05d}-of-00020.json.gz" for index in range(10, 20)
    ]
    assert {source_id: source["revision"] for source_id, source in sources.items()} == {
        "fineweb_edu": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        "dclm_edu": "dbad8ad71224482740cd9c9d353591adbf62fe04",
        "ultra_fineweb": "02c85641e3d19a854be2e09139c25adaa9518063",
        "dclm": "817d6752765f6a41261085171dd546b104f60626",
        "finemath_4plus": "e92b25a616738fe95dc186b64dfb19f9c8525594",
        "math_textbook_exercise": "fe10db8efd35597fd7fcff8ff576b5ec4ea5ff87",
        "math_multi_style": "fe10db8efd35597fd7fcff8ff576b5ec4ea5ff87",
        "wikimedia": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
        "pes2o": "636a503e44a3ca1b58e01fb61eab0825cd574de0",
        "ufw_l3_multi_style": "bc3b1ba986fcaef6871b9790a413b16267c2de0f",
        "cosmopedia_v2": "3ba9d605774198c5868892d7a8deda78031a781f",
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
