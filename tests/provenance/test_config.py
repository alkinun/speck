import json
from pathlib import Path

import pytest

from speck.config import load_experiment
from speck.data.dataset import validate_data_settings
from speck.data.loader import source_selection_counts


def test_load_experiment(tmp_path):
    (tmp_path / "model.json").write_text(json.dumps({"hidden_size": 16}))
    assert load_experiment(tmp_path, "model") == {"model": {"hidden_size": 16}}


def test_load_experiment_requires_objects(tmp_path):
    (tmp_path / "model.json").write_text("[]")
    with pytest.raises(ValueError, match="must be an object"):
        load_experiment(tmp_path, "model")


EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_extends_chains_resolve_relative_to_each_file_and_child_values_win(tmp_path):
    _write(tmp_path / "base/train.json", {"lr": 1, "run": "base", "seed": 7})
    _write(tmp_path / "sweep/train.json", {"extends": "../base/train.json", "run": "sweep"})
    _write(tmp_path / "sweep/arm/train.json", {"extends": "../train.json", "lr": 2})

    assert load_experiment(tmp_path / "sweep/arm", "train") == {
        "train": {"lr": 2, "run": "sweep", "seed": 7}
    }


def test_extends_rejects_cycles_missing_parents_and_non_paths(tmp_path):
    _write(tmp_path / "a/model.json", {"extends": "../b/model.json"})
    _write(tmp_path / "b/model.json", {"extends": "../a/model.json"})
    with pytest.raises(ValueError, match="cyclic"):
        load_experiment(tmp_path / "a", "model")

    _write(tmp_path / "c/model.json", {"extends": "missing.json"})
    with pytest.raises(FileNotFoundError, match="missing experiment config"):
        load_experiment(tmp_path / "c", "model")

    _write(tmp_path / "d/model.json", {"extends": 3})
    with pytest.raises(ValueError, match="extends must be a path string"):
        load_experiment(tmp_path / "d", "model")


def test_runtime_overlay_only_sets_device_batch_size(tmp_path):
    _write(tmp_path / "train.json", {"device_batch_size": 8, "lr": 1})
    _write(tmp_path / "runtime.json", {"device_batch_size": 2})
    assert load_experiment(tmp_path, "train")["train"] == {"device_batch_size": 2, "lr": 1}

    _write(tmp_path / "runtime.json", {"device_batch_size": 2, "lr": 3})
    with pytest.raises(ValueError, match="exactly device_batch_size"):
        load_experiment(tmp_path, "train")


def test_ladder_sweep_arm_only_changes_its_training_arm():
    names = ("data", "model", "tokenizer", "train")
    sweep = EXPERIMENTS / "ladder/50m/lr-sweep"
    arm = load_experiment(sweep / "lr-2e-3", *names)
    base = load_experiment(EXPERIMENTS / "ladder/50m", "model", "tokenizer", "train")

    assert arm["model"] == base["model"]
    assert arm["tokenizer"] == load_experiment(sweep, "tokenizer")["tokenizer"]
    assert arm["data"] == load_experiment(sweep, "data")["data"]
    assert arm["train"] == {
        **base["train"],
        "lr": 0.002,
        "run": "ladder-50m-lr-2e-3",
        "train_tokens": 658_636_800,
    }


def test_pilot_mixture_fits_quotas_at_every_world_size():
    config = load_experiment(EXPERIMENTS / "pilot", "data", "train")
    data, train = dict(config["data"]), config["train"]
    for key in ("output_name", "output_dir", "seed"):
        data.pop(key, None)
    validated = validate_data_settings(**data)
    assert sum(validated["quotas"].values()) == data["requested_train_tokens"]

    schedule = {
        "requested_train_tokens": data["requested_train_tokens"],
        "mixture": {"phases": validated["phases"]},
        "sources": [{"id": source["id"]} for source in validated["sources"]],
    }
    batch_tokens = train["batch_tokens"]
    consumed_tokens = (train["train_tokens"] + batch_tokens - 1) // batch_tokens * batch_tokens
    for world_size in (1, 2, 4, 8):
        device_batch_size = min(
            train["device_batch_size"],
            batch_tokens // (train["sequence_length"] * world_size),
        )
        stride = device_batch_size * train["sequence_length"] * world_size
        counts = source_selection_counts(schedule, "train", consumed_tokens, stride)
        for source_id, count in counts.items():
            assert count * stride + 1 <= (
                validated["quotas"][source_id] + validated["train_reserve_tokens_per_source"]
            )
