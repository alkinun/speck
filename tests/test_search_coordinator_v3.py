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
from speck.search.checkpoints import save_run_checkpoint
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
        quality_tokens_per_cost=2.0,
        evaluation_tokens_per_cost=8.0,
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
        quality_tokens_per_cost=2.0,
        evaluation_tokens_per_cost=8.0,
        profile_cost=1.0,
    )
    assert second["scheduled_actions"] == []
    assert [item["digest"] for item in study.architectures()] == architecture_digests
    assert len(study.runs()) == 3
    study.close()


def test_coordinator_selects_and_advances_posterior_anchors(tmp_path):
    digest = hashlib.sha256(b"monitor").hexdigest()
    plan = SegmentPlan(
        "dataset",
        42,
        (SegmentPartition("monitor", "val", (TokenSpan(digest, 0, 9),)),),
    )
    plan_path = tmp_path / "segments.json"
    plan_path.write_text(json.dumps(plan.export()), encoding="utf-8")
    value = settings(plan_path, plan.digest).export()
    value["calibration"].update(
        broad_architectures=3,
        noise_architectures=1,
        anchor_architectures=1,
        initialization_seeds=2,
    )
    value["planner"]["max_actions_per_event"] = 10
    for objectives in value["objective_sets"]:
        objectives["objectives"] = objectives["objectives"][:1]
    value["profiles"][0]["name"] = "cpu_reporting"
    value["profiles"][1]["name"] = "gpu_reporting"
    parsed = V3SearchSettings.from_dict(value)
    protocol = parsed.quality.resolve("dataset", "tokenizer", plan.digest)
    baseline = architecture()
    artifact_root = tmp_path / "artifacts"
    artifacts = ArtifactStore(artifact_root)
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
        artifacts=(segment_artifact,),
    )
    initial = coordinate_bootstrap(
        study,
        parsed,
        quality_tokens_per_cost=2.0,
        evaluation_tokens_per_cost=8.0,
        profile_cost=1.0,
        artifact_root=artifact_root,
    )
    assert len(initial["scheduled_actions"]) == 4
    while action := study.claim_action("trainer", kind="continue"):
        run = study.run(action["run_id"])
        checkpoint = save_run_checkpoint(
            artifacts,
            architecture_digest=run["architecture_digest"],
            protocol_digest=run["protocol_digest"],
            seed_bundle_digest=run["seed_bundle_digest"],
            steps=1,
            tokens=4,
            model_state={},
            optimizer_state={},
            data_state={},
        )
        study.commit_quality_checkpoint(
            action["id"], action["claim_token"], checkpoint
        )
    evaluations = coordinate_bootstrap(
        study,
        parsed,
        quality_tokens_per_cost=2.0,
        evaluation_tokens_per_cost=8.0,
        profile_cost=1.0,
        artifact_root=artifact_root,
    )
    assert len(evaluations["scheduled_actions"]) == 4
    while action := study.claim_action("evaluator", kind="evaluate"):
        artifact = artifacts.put_json(
            "quality_evaluation",
            {"run_id": action["run_id"]},
        )
        study.commit_quality_evaluation(
            action["id"],
            action["claim_token"],
            2.0 + action["run_id"] * 0.1,
            8,
            artifact,
        )
    shadow = coordinate_bootstrap(
        study,
        parsed,
        quality_tokens_per_cost=2.0,
        evaluation_tokens_per_cost=8.0,
        profile_cost=1.0,
        artifact_root=artifact_root,
    )
    assert shadow["phase"] == "anchors"
    assert shadow["posterior_report"] is not None
    assert len(shadow["anchors"]) == 1
    anchors = coordinate_bootstrap(
        study,
        parsed,
        quality_tokens_per_cost=2.0,
        evaluation_tokens_per_cost=8.0,
        profile_cost=1.0,
        artifact_root=artifact_root,
    )
    assert len(anchors["scheduled_actions"]) == 1
    action = study.action(anchors["scheduled_actions"][0])
    assert action["kind"] == "continue"
    assert action["payload"]["target_tokens"] == 8
    assert action["architecture_digest"] in set(shadow["anchors"])
    study.close()
