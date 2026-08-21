import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

import speck.search.study_v3 as study_module
from speck.architecture import (
    ArchitectureConfig,
    BlockConfig,
    BlockGroup,
    StageConfig,
    SwiGLUSpec,
)
from speck.search.artifacts import ArtifactEdge, ArtifactStore
from speck.search.checkpoints import save_run_checkpoint
from speck.search.protocol import (
    ObjectiveSet,
    ObjectiveSpec,
    SeedBundle,
    TrainingProtocol,
)
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
    )


def protocol():
    return TrainingProtocol(
        "calibration",
        "dataset",
        "tokenizer",
        "segments",
        8,
        32,
        2,
        "adamw",
        0.001,
        0.1,
        1,
        0.1,
        1.0,
        (32, 64),
    )


def objectives():
    return ObjectiveSet(
        "gpu_short",
        (
            ObjectiveSpec("quality.target_nll", "minimize", "quality"),
            ObjectiveSpec("gpu.decode", "minimize", "efficiency"),
        ),
    )


def quality_run(study):
    config = architecture()
    study.add_architecture(config)
    seeds = SeedBundle.create(7, 0)
    return config, seeds, study.add_run(config.digest, protocol(), seeds)


def test_v3_study_normalizes_runs_and_observations(tmp_path):
    study = V3Study(tmp_path / "study.sqlite3")
    assert study.initialize({"name": "test"}, {"device": "cpu"})
    assert study.add_objective_set(objectives())
    config = architecture()
    assert study.add_architecture(config, {"parameters": 100})
    run = study.add_run(config.digest, protocol(), SeedBundle.create(7, 0))
    assert study.add_run(config.digest, protocol(), SeedBundle.create(7, 0)) == run
    assert study.run(run)["protocol"] == protocol()
    observation = study.add_observation(
        config.digest,
        objectives().digest,
        "quality.target_nll",
        2.0,
        run_id=run,
        variance=0.01,
        tokens=32,
    )
    assert study.observations(config.digest)[0]["id"] == observation
    with pytest.raises(ValueError, match="not in"):
        study.add_observation(
            config.digest,
            objectives().digest,
            "unknown",
            1.0,
        )
    assert [event["sequence"] for event in study.events()] == list(
        range(1, len(study.events()) + 1)
    )
    study.close()


def test_v3_study_leases_actions_and_rejects_stale_results(tmp_path):
    study = V3Study(tmp_path / "study.sqlite3")
    study.initialize({}, {})
    low = study.add_action("profile", 1.0, 2.0, {"device": "cpu"})
    high = study.add_action("continue", 2.0, 3.0, {"tokens": 64})
    action = study.claim_action("worker")
    assert action["id"] == high
    assert study.heartbeat_action(high, action["claim_token"], 60)
    study.finish_action(high, action["claim_token"], {"tokens": 64})
    with pytest.raises(RuntimeError, match="stale"):
        study.finish_action(high, action["claim_token"], {})
    assert study.claim_action("worker")["id"] == low
    study.close()


def test_v3_study_releases_expired_action_leases(tmp_path):
    study = V3Study(tmp_path / "study.sqlite3")
    study.initialize({}, {})
    action_id = study.add_action("profile", 1.0, 2.0, {})
    study.claim_action("worker", lease_seconds=1)
    future = datetime.now(timezone.utc) + timedelta(seconds=2)
    assert study.release_expired_actions(future) == 1
    assert study.action(action_id)["status"] == "pending"
    study.close()


def test_v3_study_registers_artifact_lineage(tmp_path):
    study = V3Study(tmp_path / "study.sqlite3")
    study.initialize({}, {})
    store = ArtifactStore(tmp_path / "artifacts")
    parent = store.put_bytes("worker_input", b"input")
    child = store.put_bytes("worker_result", b"result")
    assert study.register_artifact(parent)
    assert study.register_artifact(child)
    assert study.add_artifact_edge(
        ArtifactEdge(parent.digest, child.digest, "produced")
    )
    assert not study.add_artifact_edge(
        ArtifactEdge(parent.digest, child.digest, "produced")
    )
    with pytest.raises(KeyError, match="unknown artifact"):
        study.add_artifact_edge(
            ArtifactEdge(parent.digest, "0" * 64, "produced")
        )
    study.close()


def test_v3_study_identity_is_immutable(tmp_path):
    path = tmp_path / "study.sqlite3"
    study = V3Study(path)
    study.initialize({"name": "first"}, {})
    study.close()
    resumed = V3Study(path)
    with pytest.raises(ValueError, match="identity changed"):
        resumed.initialize({"name": "second"}, {})
    resumed.close()


def test_v3_study_bundle_initialization_rolls_back_as_one_transaction(tmp_path):
    study = V3Study(tmp_path / "study.sqlite3")
    with pytest.raises(TypeError, match="objective set"):
        study.initialize_bundle({}, {}, objective_sets=(object(),))
    with pytest.raises(ValueError, match="not initialized"):
        study.study()
    assert study.events() == []
    study.close()


def test_v3_study_rejects_other_databases_without_modifying_them(tmp_path):
    path = tmp_path / "v2.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute("create table metadata (key text primary key, value text)")
    connection.execute("insert into metadata values ('schema_version', '2')")
    connection.commit()
    connection.close()
    before = path.read_bytes()
    with pytest.raises(ValueError, match="unsupported v3 study"):
        V3Study(path)
    assert path.read_bytes() == before


def test_v3_study_commits_quality_checkpoints_atomically(tmp_path):
    study = V3Study(tmp_path / "study.sqlite3")
    study.initialize({}, {})
    study.add_objective_set(objectives())
    config, seeds, run_id = quality_run(study)
    store = ArtifactStore(tmp_path / "artifacts")

    first_action = study.add_quality_action(run_id, 2.0, 3.0)
    study.add_action("profile", 3.0, 1.0, {})
    claim = study.claim_action("trainer", kind="continue")
    assert claim["id"] == first_action
    assert study.run(run_id)["status"] == "running"
    first = save_run_checkpoint(
        store,
        architecture_digest=config.digest,
        protocol_digest=protocol().digest,
        seed_bundle_digest=seeds.digest,
        steps=1,
        tokens=32,
        model_state={},
        optimizer_state={},
        data_state={"offset": 32},
    )
    assert study.commit_quality_checkpoint(
        first_action, claim["claim_token"], first
    ) == "paused"
    assert study.run(run_id)["checkpoint_digest"] == first.artifact.digest
    assert study.checkpoint(first.artifact.digest) == first
    assert study.checkpoints(run_id) == (first,)

    evaluation_action = study.add_evaluation_action(
        run_id,
        (objectives().digest,),
        10,
        1.0,
        1.0,
    )
    evaluation_claim = study.claim_action("evaluator", kind="evaluate")
    evaluation_artifact = store.put_json("quality_evaluation", {"nll": 2.0})
    study.commit_quality_evaluation(
        evaluation_action,
        evaluation_claim["claim_token"],
        2.0,
        10,
        evaluation_artifact,
    )

    second_action = study.add_quality_action(run_id, 2.0, 3.0)
    second_claim = study.claim_action("trainer", kind="continue")
    second = save_run_checkpoint(
        store,
        architecture_digest=config.digest,
        protocol_digest=protocol().digest,
        seed_bundle_digest=seeds.digest,
        steps=2,
        tokens=64,
        model_state={},
        optimizer_state={},
        data_state={"offset": 64},
        parent=first,
    )
    assert study.commit_quality_checkpoint(
        second_action, second_claim["claim_token"], second
    ) == "completed"
    assert study.run(run_id)["status"] == "completed"
    assert study.architecture(config.digest)["config"].digest == config.digest
    second_evaluation = study.add_evaluation_action(
        run_id,
        (objectives().digest,),
        10,
        1.0,
        1.0,
    )
    second_evaluation_claim = study.claim_action("evaluator", kind="evaluate")
    second_evaluation_artifact = store.put_json(
        "quality_evaluation", {"nll": 1.5}
    )
    study.commit_quality_evaluation(
        second_evaluation,
        second_evaluation_claim["claim_token"],
        1.5,
        10,
        second_evaluation_artifact,
    )
    assert study.prune_checkpoint_payload(
        run_id,
        first.artifact.digest,
        store,
        "superseded",
    )
    assert not store.path(first.artifact).exists()
    assert study.checkpoint_payload_pruned(first.artifact.digest)
    assert study.checkpoint(first.artifact.digest) == first
    assert study.prune_checkpoint_payload(
        run_id,
        second.artifact.digest,
        store,
        "trajectory_complete",
        archive_run=True,
    )
    assert study.run(run_id)["status"] == "archived"
    assert not store.path(second.artifact).exists()
    study.close()


def test_v3_study_rolls_back_invalid_quality_checkpoint(tmp_path):
    study = V3Study(tmp_path / "study.sqlite3")
    study.initialize({}, {})
    _, seeds, run_id = quality_run(study)
    action_id = study.add_quality_action(run_id, 1.0, 1.0)
    claim = study.claim_action("trainer", kind="continue")
    checkpoint = save_run_checkpoint(
        ArtifactStore(tmp_path / "artifacts"),
        architecture_digest="different",
        protocol_digest=protocol().digest,
        seed_bundle_digest=seeds.digest,
        steps=1,
        tokens=32,
        model_state={},
        optimizer_state={},
        data_state={},
    )
    with pytest.raises(ValueError, match="identity"):
        study.commit_quality_checkpoint(action_id, claim["claim_token"], checkpoint)
    with pytest.raises(KeyError):
        study.artifact(checkpoint.artifact.digest)
    assert study.action(action_id)["status"] == "running"
    assert study.run(run_id)["tokens"] == 0
    study.finish_action(action_id, claim["claim_token"], error="invalid checkpoint")
    assert study.run(run_id)["status"] == "pending"
    study.close()


def test_v3_study_rejects_work_after_lease_expiration(tmp_path, monkeypatch):
    current = datetime(2026, 1, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(study_module, "_now", lambda: current)
    study = V3Study(tmp_path / "study.sqlite3")
    study.initialize({}, {})
    _, _, run_id = quality_run(study)
    action_id = study.add_quality_action(run_id, 1.0, 1.0)
    claim = study.claim_action("trainer", lease_seconds=1, kind="continue")
    current += timedelta(seconds=2)
    with pytest.raises(RuntimeError, match="stale"):
        study.heartbeat_action(action_id, claim["claim_token"])
    with pytest.raises(RuntimeError, match="stale"):
        study.finish_action(action_id, claim["claim_token"], error="late")
    assert study.release_expired_actions(current) == 1
    assert study.run(run_id)["status"] == "pending"
    study.close()


def test_v3_study_allows_one_active_quality_action_per_run(tmp_path):
    study = V3Study(tmp_path / "study.sqlite3")
    study.initialize({}, {})
    _, _, run_id = quality_run(study)
    study.add_quality_action(run_id, 1.0, 1.0)
    with pytest.raises(ValueError, match="active action"):
        study.add_quality_action(run_id, 1.0, 1.0)
    study.close()
