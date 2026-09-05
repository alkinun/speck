import json
import shutil
from pathlib import Path

import pytest

from speck.config import load_experiment
from speck.paper_baseline import materialize_baselines, mixer_counts

root = Path(__file__).parents[1]
matrix = root / "research" / "paper-1" / "baseline_matrix.json"
materialized = root / "experiments" / "Speck-Paper1-Baselines-131M"


def declared_geometry(model):
    kinds = mixer_counts(model)
    flops = 1_301_237_760 if kinds == {"attention_global": 20} else 1_021_601_280
    return {
        "parameters": model["expected_parameters"],
        "flops_per_token_at_4096": flops,
    }


def test_checked_paper_baseline_materialization_matches_contract():
    result = materialize_baselines(matrix, check=True, geometry_fn=declared_geometry)

    assert result["status"] == "materialized_unexecuted"
    assert set(result["arms"]) == {"dense_global_param_match", "five_cache_kda_gqa"}
    assert [pair["seed"] for pair in result["pairs"]] == [42, 43, 44]
    assert len(result["generated_files"]) == 48


def test_materialized_pairs_resolve_distinct_data_orders_and_shared_recipe():
    pair0 = materialized / "runs" / "pair-0-seed-42-order-0"
    pair1 = materialized / "runs" / "pair-1-seed-43-order-536870912"
    dense0 = load_experiment(pair0 / "dense_global_param_match", "model", "train")
    kda0 = load_experiment(pair0 / "five_cache_kda_gqa", "model", "train")
    dense1 = load_experiment(pair1 / "dense_global_param_match", "model", "train")

    assert dense0["model"]["expected_parameters"] == 153_977_088
    assert kda0["model"]["expected_parameters"] == 153_958_938
    assert dense0["train"]["data_token_offset"] == 0
    assert dense1["train"]["data_token_offset"] == 536_870_912
    assert dense1["train"]["seed"] == 43
    ignored = {"data_token_offset", "run", "seed"}
    assert {key: value for key, value in dense0["train"].items() if key not in ignored} == {
        key: value for key, value in kda0["train"].items() if key not in ignored
    }


def test_materialization_check_rejects_generated_config_drift(tmp_path):
    copied = tmp_path / "baseline"
    shutil.copytree(materialized, copied)
    path = copied / "runs" / "pair-0-seed-42-order-0" / "dense_global_param_match" / "train.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["seed"] = 99
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="materialization drift"):
        materialize_baselines(
            matrix,
            output_root=copied,
            check=True,
            geometry_fn=declared_geometry,
        )
