import hashlib
import json

from speck.architecture import (
    ArchitectureConfig,
    BlockConfig,
    BlockGroup,
    StageConfig,
    SwiGLUSpec,
)
from speck.search.artifacts import ArtifactStore
from speck.search.coordinator_v3 import coordinate_bootstrap
from speck.search.segments import SegmentPartition, SegmentPlan, TokenSpan
from speck.search.spec_v3 import V3SearchSettings
from speck.search.study_v3 import V3Study


def architecture():
    return ArchitectureConfig(
        (
            BlockGroup(
                BlockConfig(8, (StageConfig((SwiGLUSpec(16),)),))
            ),
        ),
        8,
        vocab_size=16,
        max_position_embeddings=16,
    )


def settings(plan_path, plan_digest):
    return V3SearchSettings.from_dict(
        {
            "format_version": 3,
            "seed": 42,
            "segment_plan": {
                "path": str(plan_path),
                "expected_digest": plan_digest,
            },
            "quality": {
                "name": "calibration",
                "sequence_length": 4,
                "batch_tokens": 4,
                "device_batch_size": 1,
                "optimizer": "adamw",
                "learning_rate": 0.001,
                "minimum_learning_rate_scale": 0.1,
                "warmup_steps": 0,
                "weight_decay": 0.1,
                "gradient_clip": 1.0,
                "checkpoint_tokens": [4, 8],
                "device_type": "cpu",
            },
            "calibration": {
                "noise_architectures": 1,
                "broad_architectures": 2,
                "anchor_architectures": 1,
                "initialization_seeds": 2,
                "data_seeds": 1,
                "numerical_repeats": 1,
                "noise_tokens": 4,
                "broad_tokens": 4,
                "anchor_tokens": 8,
                "bootstrap_samples": 10,
            },
            "planner": {
                "total_cost": 100.0,
                "cost_unit": "wall_seconds",
                "max_actions_per_event": 2,
                "posterior_samples": 10,
                "surrogate_models": 2,
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
                    "name": "cpu_short",
                    "objectives": [
                        {
                            "name": "quality.target_nll",
                            "direction": "minimize",
                            "role": "quality",
                        },
                        {
                            "name": "cpu_short.prefill_p95",
                            "direction": "minimize",
                            "role": "efficiency",
                        },
                    ],
                },
                {
                    "name": "gpu_short",
                    "objectives": [
                        {
                            "name": "quality.target_nll",
                            "direction": "minimize",
                            "role": "quality",
                        },
                        {
                            "name": "gpu_short.prefill_p95",
                            "direction": "minimize",
                            "role": "efficiency",
                        },
                    ],
                },
            ],
            "profiles": [
                {
                    "name": "cpu_short",
                    "backend": "torch_native",
                    "device": "cpu",
                    "dtype": "float32",
                    "cache_dtype": "float32",
                    "batch_size": 1,
                    "prompt_tokens": 4,
                    "generated_tokens": 2,
                    "warmup_requests": 0,
                    "measured_requests": 2,
                    "process_repetitions": 1,
                },
                {
                    "name": "gpu_short",
                    "backend": "torch_native",
                    "device": "cuda",
                    "dtype": "float32",
                    "cache_dtype": "float32",
                    "batch_size": 1,
                    "prompt_tokens": 4,
                    "generated_tokens": 2,
                    "warmup_requests": 0,
                    "measured_requests": 2,
                    "process_repetitions": 1,
                },
            ],
        }
    )


def test_bootstrap_coordinator_builds_panel_and_respects_action_slots(tmp_path):
    digest = hashlib.sha256(b"monitor").hexdigest()
    plan = SegmentPlan(
        "dataset",
        42,
        (
            SegmentPartition(
                "monitor",
                "val",
                (TokenSpan(digest, 0, 9),),
            ),
        ),
    )
    plan_path = tmp_path / "segments.json"
    plan_path.write_text(json.dumps(plan.export()), encoding="utf-8")
    parsed = settings(plan_path, plan.digest)
    protocol = parsed.quality.resolve("dataset", "tokenizer", plan.digest)
    baseline = architecture()
    artifacts = ArtifactStore(tmp_path / "artifacts")
    segment_artifact = artifacts.put_json("segment_plan", plan.export())
    study = V3Study(tmp_path / "study.sqlite3")
    study.initialize_bundle(
        parsed.export(),
        {
            "model_digest": baseline.digest,
            "resolved_protocol": protocol.__dict__,
            "resolved_protocol_digest": protocol.digest,
            "segment_plan": {"digest": plan.digest, "path": str(plan_path)},
        },
        objective_sets=parsed.objective_sets,
        architecture=baseline,
        static={"parameters": baseline.vocab_size * baseline.embedding_size},
        operation={"operator": "baseline"},
        artifacts=(segment_artifact,),
    )
    first = coordinate_bootstrap(
        study,
        parsed,
        quality_cost=2.0,
        evaluation_cost=1.0,
        profile_cost=1.0,
    )
    assert first["phase"] == "bootstrap"
    assert first["architectures"] == 2
    assert first["runs"] == 3
    assert len(first["scheduled_actions"]) == 2
    assert len(study.actions("pending")) == 2
    architecture_digests = [item["digest"] for item in study.architectures()]

    second = coordinate_bootstrap(
        study,
        parsed,
        quality_cost=2.0,
        evaluation_cost=1.0,
        profile_cost=1.0,
    )
    assert second["scheduled_actions"] == []
    assert [item["digest"] for item in study.architectures()] == architecture_digests
    assert len(study.runs()) == 3
    study.close()
