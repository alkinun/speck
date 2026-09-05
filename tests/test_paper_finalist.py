import json
from pathlib import Path

import pytest

from speck.config import load_experiment
from speck.paper_finalist import materialize_finalist

root = Path(__file__).parents[1]
contract = root / "research" / "paper-1" / "finalist_materialization_v1.json"


def test_finalist_materializes_exact_six_pair_crossing(tmp_path):
    output = tmp_path / "finalist"
    manifest = materialize_finalist(contract, output)

    assert manifest["status"] == "materialized_unexecuted_training_blocked"
    assert not manifest["training_authorized"]
    assert len(manifest["pairs"]) == 6
    assert len(manifest["generated_files"]) == 84
    assert len(manifest["generated_config_sha256"]) == 84
    assert set(manifest["arms"]) == {
        "dense_global_param_match",
        "five_cache_kda_gqa",
    }
    assert materialize_finalist(contract, output, check=True) == manifest


def test_finalist_run_inherits_recipe_and_changes_only_frozen_horizon(tmp_path):
    output = tmp_path / "finalist"
    materialize_finalist(contract, output)
    pair = output / "runs" / "pair-3-seed-43-order-1610612736"
    dense = load_experiment(pair / "dense_global_param_match", "model", "runtime", "train")
    candidate = load_experiment(pair / "five_cache_kda_gqa", "model", "runtime", "train")

    assert dense["train"]["seed"] == candidate["train"]["seed"] == 43
    assert dense["train"]["data_token_offset"] == 1_610_612_736
    assert dense["train"]["train_tokens"] == candidate["train"]["train_tokens"] == 1_539_833_856
    assert dense["train"]["save_every"] == candidate["train"]["save_every"] == 23_496
    assert dense["train"]["eval_every"] == candidate["train"]["eval_every"] == 5_874
    assert dense["runtime"] == candidate["runtime"] == {"device_batch_size": 4}
    assert dense["model"]["expected_parameters"] == 153_977_088
    assert candidate["model"]["expected_parameters"] == 153_958_938
    assert dense["train"]["output_dir"] != candidate["train"]["output_dir"]


def test_finalist_refuses_overwrite(tmp_path):
    output = tmp_path / "finalist"
    materialize_finalist(contract, output)
    with pytest.raises(FileExistsError, match="already exists"):
        materialize_finalist(contract, output)


def test_finalist_check_detects_config_drift(tmp_path):
    output = tmp_path / "finalist"
    materialize_finalist(contract, output)
    path = (
        output
        / "runs"
        / "pair-0-seed-42-order-0"
        / "dense_global_param_match"
        / "train.json"
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    value["seed"] = 99
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="materialization drift"):
        materialize_finalist(contract, output, check=True)
