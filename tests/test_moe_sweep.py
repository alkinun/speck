import json
import math
from pathlib import Path

import torch

from speck.architecture import ArchitectureConfig, RoutedSwiGLUSpec, SwiGLUSpec
from speck.config import load_experiment
from speck.dataset import validate_data_settings
from speck.model import build_model
from speck.train import checkpoint_milestones

ROOT = Path(__file__).parents[1]
ARMS = ("D0", "M1", "M2", "M3")
EXPECTED = {
    "D0": (140_652_288, 140_652_288, 16),
    "M1": (411_485_952, 140_756_736, 8),
    "M2": (411_590_400, 140_861_184, 8),
    "M3": (772_771_584, 141_070_080, 2),
}


def experiment(arm):
    return ROOT / "experiments" / f"SpeckLabs-1B-{arm}"


def test_sweep_shares_byte_identical_input_and_training_recipes():
    for name in ("data.json", "tokenizer.json", "train.json"):
        payloads = [(experiment(arm) / name).read_bytes() for arm in ARMS]
        assert len(set(payloads)) == 1

    resolved = [
        load_experiment(experiment(arm), "data", "tokenizer", "train") for arm in ARMS
    ]
    assert all(item["data"] == resolved[0]["data"] for item in resolved)
    assert all(item["tokenizer"] == resolved[0]["tokenizer"] for item in resolved)
    for item in resolved:
        item["train"].pop("device_batch_size")
    assert all(item["train"] == resolved[0]["train"] for item in resolved)


def test_sweep_models_have_exact_total_and_active_parameter_counts():
    for arm, (total, active, _) in EXPECTED.items():
        settings = load_experiment(experiment(arm), "model")["model"]
        with torch.device("meta"):
            model = build_model(settings, vocab_size=32_000)
        assert model.parameter_count() == total
        assert model.active_parameter_count() == active


def test_sweep_keeps_the_hybrid_backbone_and_routes_after_the_first_ffn():
    dense = ArchitectureConfig.from_dict(
        load_experiment(experiment("D0"), "model")["model"]
    )
    dense_mixers = [
        invocation.block.stages[0].branches[0]
        for invocation in dense.execution_plan
    ]
    for arm in ARMS[1:]:
        routed = ArchitectureConfig.from_dict(
            load_experiment(experiment(arm), "model")["model"]
        )
        mixers = [
            invocation.block.stages[0].branches[0]
            for invocation in routed.execution_plan
        ]
        assert mixers == dense_mixers
        feed_forwards = [
            invocation.block.stages[1].branches[0]
            for invocation in routed.execution_plan
        ]
        assert isinstance(feed_forwards[0], SwiGLUSpec)
        assert all(isinstance(operation, RoutedSwiGLUSpec) for operation in feed_forwards[1:])


def test_sweep_schedule_and_calibrated_accumulation_are_exact():
    for arm, (_, _, device_batch) in EXPECTED.items():
        training = load_experiment(experiment(arm), "train")["train"]
        assert training["device_batch_size"] == device_batch
        assert math.ceil(training["train_tokens"] / training["batch_tokens"]) == 15_259
        assert 15_259 * training["batch_tokens"] == 1_000_013_824
        assert training["batch_tokens"] // (
            device_batch * training["sequence_length"]
        ) == {"D0": 2, "M1": 4, "M2": 4, "M3": 16}[arm]
        assert checkpoint_milestones(
            training["checkpoint_tokens"], training["batch_tokens"], 0, 15_259
        ) == {763: 50_000_000, 7_630: 500_000_000, 15_259: 1_000_000_000}


def test_sweep_corpus_is_one_pinned_filtered_dclm_edu_source():
    data = load_experiment(experiment("D0"), "data")["data"]
    source = data["sources"][0]
    assert source["repo"] == "HuggingFaceTB/dclm-edu"
    assert source["revision"] == "dbad8ad71224482740cd9c9d353591adbf62fe04"
    assert source["filters"] == {
        "language": "en",
        "min_score": 3,
        "score_operator": ">=",
    }
    assert data["requested_train_tokens"] == 1_000_000_000
    assert data["validation_tokens_per_source"] == 20_000_000
    settings = validate_data_settings(
        **{
            key: data[key]
            for key in (
                "sources",
                "mixture",
                "requested_train_tokens",
                "validation_tokens_per_source",
                "validation_fraction",
                "filtering",
                "dedup",
                "shards",
            )
        }
    )
    assert settings["quotas"] == {"dclm_edu": 1_000_000_000}


def test_qualification_report_matches_committed_runtime_ceilings():
    report = json.loads(
        (ROOT / "experiments/SpeckLabs-1B-shared/qualification.json").read_text()
    )
    assert report["device"] == "NVIDIA GeForce RTX 3090"
    assert report["limit_mib"] == 0.9 * 24_576
    for arm, (_, _, device_batch) in EXPECTED.items():
        result = report["results"][arm]
        assert result["selected_device_batch_size"] == device_batch
        assert result["selected_peak_reserved_mib"] <= report["limit_mib"]
