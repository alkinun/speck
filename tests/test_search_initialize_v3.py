import json

import pytest

import speck.dataset as dataset
from speck.search.initialize_v3 import initialize_study
from speck.search.segments import build_segment_plan_from_dataset
from speck.search.spec_v3 import V3SearchSettings
from speck.search.study_v3 import V3Study


class FakeTokenizer:
    vocab_size = 32
    bos_id = 1
    eos_id = 2

    def encode_batch(self, texts, bos=False, eos=False):
        return [
            ([self.bos_id] if bos else [])
            + [3 + byte % 29 for byte in text.encode()]
            + ([self.eos_id] if eos else [])
            for text in texts
        ]

    def fingerprint(self):
        return "1" * 64


def search_settings(segment_path, segment_digest):
    return V3SearchSettings.from_dict(
        {
            "format_version": 3,
            "seed": 42,
            "segment_plan": {
                "path": str(segment_path),
                "expected_digest": segment_digest,
            },
            "quality": {
                "name": "calibration",
                "sequence_length": 8,
                "batch_tokens": 8,
                "device_batch_size": 1,
                "optimizer": "adamw",
                "learning_rate": 0.001,
                "minimum_learning_rate_scale": 0.1,
                "warmup_steps": 1,
                "weight_decay": 0.1,
                "gradient_clip": 1.0,
                "checkpoint_tokens": [8, 16],
            },
            "calibration": {
                "noise_architectures": 1,
                "broad_architectures": 2,
                "anchor_architectures": 1,
                "initialization_seeds": 1,
                "data_seeds": 1,
                "numerical_repeats": 2,
                "noise_tokens": 8,
                "broad_tokens": 8,
                "anchor_tokens": 16,
                "bootstrap_samples": 10,
            },
            "planner": {
                "total_cost": 100.0,
                "cost_unit": "wall_seconds",
                "max_actions_per_event": 1,
                "posterior_samples": 10,
                "surrogate_models": 2,
                "surrogate_ridge": 0.001,
            },
            "space": {
                "min_logical_depth": 1,
                "max_logical_depth": 2,
                "hidden_sizes": [8],
                "intermediate_sizes": [16],
                "head_dims": [4],
                "kv_heads": [1],
                "sliding_windows": [4],
                "conv_kernel_sizes": [3],
                "conv_inner_sizes": [8],
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
                        }
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
    )


def prepared_inputs(tmp_path, monkeypatch):
    tokenizer = FakeTokenizer()
    monkeypatch.setattr(dataset, "get_tokenizer", lambda: tokenizer)
    monkeypatch.setattr(
        dataset,
        "_is_validation_document",
        lambda content, seed, fraction: content.startswith("validation"),
    )
    documents = []
    for index in range(20):
        documents.extend(
            (
                {"content": f"validation-{index}", "source": "test", "score": 1.0},
                {"content": f"training-{index}", "source": "test", "score": 1.0},
            )
        )
    data_dir = tmp_path / "packed"
    dataset.prepare_dataset(
        train_tokens=100,
        validation_tokens=100,
        shard_tokens=37,
        output_dir=data_dir,
        document_iterator=iter(documents),
        tokenizer=tokenizer,
    )
    plan = build_segment_plan_from_dataset(
        data_dir,
        42,
        20,
        {"monitor": 1, "promotion": 1, "audit": 1, "final": 1},
    )
    segment_path = tmp_path / "segments.json"
    segment_path.write_text(json.dumps(plan.export()), encoding="utf-8")
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    for name in ("data", "model", "tokenizer"):
        (experiment / f"{name}.json").write_text("{}", encoding="utf-8")
    model_settings = {
        "architecture": "speck",
        "head_dim": 4,
        "layers": [
            {
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_key_value_heads": 1,
            }
        ],
        "max_position_embeddings": 16,
    }
    return {
        "data_dir": data_dir,
        "experiment": experiment,
        "model_settings": model_settings,
        "plan": plan,
        "segment_path": segment_path,
        "tokenizer": tokenizer,
    }


def test_v3_initialization_registers_a_validated_baseline_bundle(
    tmp_path, monkeypatch
):
    values = prepared_inputs(tmp_path, monkeypatch)
    settings = search_settings(values["segment_path"], values["plan"].digest)
    study_path = tmp_path / "study.sqlite3"
    arguments = {
        "experiment": values["experiment"],
        "model_settings": values["model_settings"],
        "tokenizer_settings": {"repo": "test"},
        "data_settings": {"output_dir": str(values["data_dir"])},
        "tokenizer": values["tokenizer"],
        "captured_git": {
            "dirty": False,
            "revision": "test",
            "working_tree": "0" * 64,
        },
        "environment": {"device": "test"},
    }
    result = initialize_study(
        study_path,
        tmp_path / "artifacts",
        settings,
        **arguments,
    )
    assert result["initialized"]
    assert result["evaluation_tokens"] == next(
        partition.tokens - 1
        for partition in values["plan"].partitions
        if partition.name == "monitor"
    )
    study = V3Study(study_path, readonly=True)
    assert study.study()["provenance"]["resolved_protocol_digest"] == result[
        "protocol_digest"
    ]
    assert study.objective_set(settings.objective_sets[0].digest) == settings.objective_sets[0]
    assert study.architecture(result["architecture_digest"])["static"]["parameters"] > 0
    assert study.artifact(values["plan"].digest).kind == "segment_plan"
    event_count = len(study.events())
    study.close()

    repeated = initialize_study(
        study_path,
        tmp_path / "artifacts",
        settings,
        **arguments,
    )
    assert not repeated["initialized"]
    study = V3Study(study_path, readonly=True)
    assert len(study.events()) == event_count
    study.close()


def test_v3_initialization_rejects_an_unfrozen_segment_plan(tmp_path, monkeypatch):
    values = prepared_inputs(tmp_path, monkeypatch)
    settings = search_settings(values["segment_path"], None)
    with pytest.raises(ValueError, match="frozen segment plan"):
        initialize_study(
            tmp_path / "study.sqlite3",
            tmp_path / "artifacts",
            settings,
            experiment=values["experiment"],
            model_settings=values["model_settings"],
            tokenizer_settings={},
            data_settings={"output_dir": str(values["data_dir"])},
            tokenizer=values["tokenizer"],
        )
    assert not (tmp_path / "study.sqlite3").exists()
