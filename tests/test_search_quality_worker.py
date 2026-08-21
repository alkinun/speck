from dataclasses import replace
import json

import pytest
import torch

import speck.dataset as dataset
from speck.architecture import (
    ArchitectureConfig,
    BlockConfig,
    BlockGroup,
    StageConfig,
    SwiGLUSpec,
)
from speck.dataloader import manifest_fingerprint
from speck.search.artifacts import ArtifactStore
from speck.search.checkpoints import load_run_checkpoint
from speck.search.evaluation_worker import run_evaluation_worker
from speck.search.protocol import (
    ObjectiveSet,
    ObjectiveSpec,
    SeedBundle,
    TrainingProtocol,
)
from speck.search.quality_worker import run_quality_worker
from speck.search.segments import (
    SegmentPartition,
    SegmentPlan,
    TokenSpan,
    load_document_index,
)
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


def quality_data(tmp_path, monkeypatch):
    tokenizer = FakeTokenizer()
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
    manifest = dataset.prepare_dataset(
        train_tokens=100,
        validation_tokens=20,
        shard_tokens=31,
        output_dir=data_dir,
        document_iterator=iter(documents),
        tokenizer=tokenizer,
    )
    records = load_document_index(data_dir, manifest)
    training = tuple(record for record in records if record.split == "train")
    monitor = tuple(record for record in records if record.split == "val")
    plan = SegmentPlan(
        manifest_fingerprint(manifest),
        42,
        (
            SegmentPartition(
                "train",
                "train",
                tuple(
                    TokenSpan(record.content_hash, record.start_token, record.end_token)
                    for record in training
                ),
            ),
            SegmentPartition(
                "monitor",
                "val",
                tuple(
                    TokenSpan(record.content_hash, record.start_token, record.end_token)
                    for record in monitor
                ),
            ),
        ),
    )
    plan_path = tmp_path / "segments.json"
    plan_path.write_text(json.dumps(plan.export()), encoding="utf-8")
    return data_dir, plan, plan_path, tokenizer


def architecture():
    return ArchitectureConfig(
        (
            BlockGroup(
                BlockConfig(8, (StageConfig((SwiGLUSpec(16),)),))
            ),
        ),
        8,
        vocab_size=32,
        max_position_embeddings=8,
    )


def protocol(plan, checkpoints):
    return TrainingProtocol(
        "calibration",
        plan.dataset_digest,
        "1" * 64,
        plan.digest,
        4,
        4,
        1,
        "adamw",
        0.001,
        0.1,
        0,
        0.1,
        1.0,
        checkpoints,
        device_type="cpu",
    )


def quality_objectives():
    return ObjectiveSet(
        "cpu_short",
        (
            ObjectiveSpec("quality.target_nll", "minimize", "quality"),
            ObjectiveSpec(
                "quality.procedural_score",
                "maximize",
                "reporting",
                required_for_selection=False,
            ),
        ),
    )


def prepare_study(path, artifact_root, data_dir, plan, plan_path, run_protocol):
    artifacts = ArtifactStore(artifact_root)
    segment_artifact = artifacts.put_json("segment_plan", plan.export())
    git = {"dirty": False, "revision": "test", "working_tree": "0" * 64}
    study = V3Study(path)
    study.initialize_bundle(
        {"seed": 42},
        {
            "dataset_dir": str(data_dir),
            "git": git,
            "segment_plan": {"digest": plan.digest, "path": str(plan_path)},
            "tokenizer": {},
        },
        objective_sets=(quality_objectives(),),
        architecture=architecture(),
        artifacts=(segment_artifact,),
    )
    seeds = SeedBundle.create(42, 0)
    run_id = study.add_run(architecture().digest, run_protocol, seeds)
    study.close()
    return git, run_id


def model_state(artifact_root, study_path, run_id):
    study = V3Study(study_path, readonly=True)
    checkpoint = study.checkpoint(study.run(run_id)["checkpoint_digest"])
    study.close()
    return load_run_checkpoint(ArtifactStore(artifact_root), checkpoint)["model"]


def test_quality_worker_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    data_dir, plan, plan_path, tokenizer = quality_data(tmp_path, monkeypatch)
    resumed_path = tmp_path / "resumed.sqlite3"
    resumed_artifacts = tmp_path / "resumed-artifacts"
    git, run_id = prepare_study(
        resumed_path,
        resumed_artifacts,
        data_dir,
        plan,
        plan_path,
        protocol(plan, (4, 8)),
    )
    study = V3Study(resumed_path)
    study.add_quality_action(run_id, 1.0, 1.0)
    study.close()
    first = run_quality_worker(
        resumed_path,
        resumed_artifacts,
        owner="test",
        device="cpu",
        lease_seconds=30,
        tokenizer=tokenizer,
        captured_git=git,
    )
    assert first["status"] == "paused"
    monitor_tokens = next(
        partition.tokens for partition in plan.partitions if partition.name == "monitor"
    ) - 1
    study = V3Study(resumed_path)
    study.add_evaluation_action(
        run_id,
        (quality_objectives().digest,),
        monitor_tokens,
        1.0,
        1.0,
    )
    study.close()
    run_evaluation_worker(
        resumed_path,
        resumed_artifacts,
        owner="evaluator",
        device="cpu",
        tokenizer=tokenizer,
        captured_git=git,
    )
    study = V3Study(resumed_path)
    study.add_quality_action(run_id, 1.0, 1.0)
    study.close()
    second = run_quality_worker(
        resumed_path,
        resumed_artifacts,
        owner="test",
        device="cpu",
        lease_seconds=30,
        tokenizer=tokenizer,
        captured_git=git,
    )
    assert second["status"] == "completed"

    direct_path = tmp_path / "direct.sqlite3"
    direct_artifacts = tmp_path / "direct-artifacts"
    direct_git, direct_run = prepare_study(
        direct_path,
        direct_artifacts,
        data_dir,
        plan,
        plan_path,
        protocol(plan, (8,)),
    )
    study = V3Study(direct_path)
    study.add_quality_action(direct_run, 1.0, 1.0)
    study.close()
    direct = run_quality_worker(
        direct_path,
        direct_artifacts,
        owner="test",
        device="cpu",
        lease_seconds=30,
        tokenizer=tokenizer,
        captured_git=direct_git,
    )
    assert direct["status"] == "completed"

    resumed_state = model_state(resumed_artifacts, resumed_path, run_id)
    direct_state = model_state(direct_artifacts, direct_path, direct_run)
    assert resumed_state.keys() == direct_state.keys()
    for name in resumed_state:
        assert torch.equal(resumed_state[name], direct_state[name]), name
    assert run_quality_worker(
        resumed_path,
        resumed_artifacts,
        owner="test",
        device="cpu",
        tokenizer=tokenizer,
        captured_git=git,
    ) is None


def test_quality_worker_returns_failed_setup_to_a_resumable_run(tmp_path, monkeypatch):
    data_dir, plan, plan_path, tokenizer = quality_data(tmp_path, monkeypatch)
    study_path = tmp_path / "study.sqlite3"
    artifact_root = tmp_path / "artifacts"
    run_protocol = replace(protocol(plan, (4,)), compile_model=True)
    git, run_id = prepare_study(
        study_path,
        artifact_root,
        data_dir,
        plan,
        plan_path,
        run_protocol,
    )
    study = V3Study(study_path)
    action_id = study.add_quality_action(run_id, 1.0, 1.0)
    study.close()
    with pytest.raises(ValueError, match="compiled"):
        run_quality_worker(
            study_path,
            artifact_root,
            owner="test",
            device="cpu",
            tokenizer=tokenizer,
            captured_git=git,
        )
    study = V3Study(study_path, readonly=True)
    assert study.action(action_id)["status"] == "failed"
    assert study.run(run_id)["status"] == "pending"
    study.close()


def test_evaluation_worker_observes_the_whole_monitor_partition(
    tmp_path, monkeypatch
):
    data_dir, plan, plan_path, tokenizer = quality_data(tmp_path, monkeypatch)
    study_path = tmp_path / "study.sqlite3"
    artifact_root = tmp_path / "artifacts"
    git, run_id = prepare_study(
        study_path,
        artifact_root,
        data_dir,
        plan,
        plan_path,
        protocol(plan, (4,)),
    )
    study = V3Study(study_path)
    study.add_quality_action(run_id, 1.0, 1.0)
    study.close()
    run_quality_worker(
        study_path,
        artifact_root,
        owner="trainer",
        device="cpu",
        tokenizer=tokenizer,
        captured_git=git,
    )
    monitor_tokens = next(
        partition.tokens for partition in plan.partitions if partition.name == "monitor"
    ) - 1
    study = V3Study(study_path)
    action_id = study.add_evaluation_action(
        run_id,
        (quality_objectives().digest,),
        monitor_tokens,
        1.0,
        1.0,
    )
    study.close()
    result = run_evaluation_worker(
        study_path,
        artifact_root,
        owner="evaluator",
        device="cpu",
        tokenizer=tokenizer,
        captured_git=git,
    )
    assert result["action_id"] == action_id
    assert result["evaluated_tokens"] == monitor_tokens
    assert result["nll"] > 0
    study = V3Study(study_path, readonly=True)
    run = study.run(run_id)
    assert run["status"] == "completed"
    evaluation = study.quality_evaluation(run_id, run["checkpoint_digest"])
    assert evaluation["evaluated_tokens"] == monitor_tokens
    observations = study.observations(
        architecture().digest,
        quality_objectives().digest,
    )
    assert [item["objective_name"] for item in observations] == [
        "quality.target_nll"
    ]
    assert [item.name for item in quality_objectives().selection] == [
        "quality.target_nll"
    ]
    study.close()
