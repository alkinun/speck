import json

import pytest

from speck.config import load_experiment


def test_load_experiment(tmp_path):
    (tmp_path / "model.json").write_text(json.dumps({"hidden_size": 16}))
    assert load_experiment(tmp_path, "model") == {"model": {"hidden_size": 16}}


def test_load_experiment_requires_objects(tmp_path):
    (tmp_path / "model.json").write_text("[]")
    with pytest.raises(ValueError, match="must be an object"):
        load_experiment(tmp_path, "model")


def test_speck1_1_sft_experiment_uses_speckchat2_and_original_base():
    current = load_experiment("experiments/Speck1-140M", "model", "tokenizer", "sft")
    updated = load_experiment("experiments/Speck1.1-140M", "model", "tokenizer", "sft")

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
    assert updated["sft"]["epochs"] == 2
    assert updated["sft"]["run"] == "Speck1.1-140M-Instruct"


def test_speck1_1_140m_light_uses_one_epoch_without_changing_data_or_base():
    full = load_experiment("experiments/Speck1.1-140M", "model", "tokenizer", "sft")
    light = load_experiment(
        "experiments/Speck1.1-140M-Instruct-Light", "model", "tokenizer", "sft"
    )

    assert light["model"] == full["model"]
    assert light["tokenizer"] == full["tokenizer"]
    assert light["sft"]["dataset"] == full["sft"]["dataset"]
    assert light["sft"]["pretrained"] == full["sft"]["pretrained"]
    assert light["sft"]["epochs"] == 1
    assert light["sft"]["run"] == "Speck1.1-140M-Instruct-Light"
