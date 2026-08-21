from datetime import datetime, timedelta, timezone

import pytest

from speck.architecture import (
    ArchitectureConfig,
    BlockConfig,
    BlockGroup,
    StageConfig,
    SwiGLUSpec,
)
from speck.search.artifacts import ArtifactEdge, ArtifactStore
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
            ObjectiveSpec("quality.nll", "minimize", "quality"),
            ObjectiveSpec("gpu.decode", "minimize", "efficiency"),
        ),
    )


def test_v3_study_normalizes_runs_and_observations(tmp_path):
    study = V3Study(tmp_path / "study.sqlite3")
    assert study.initialize({"name": "test"}, {"device": "cpu"})
    assert study.add_objective_set(objectives())
    config = architecture()
    assert study.add_architecture(config, {"parameters": 100})
    run = study.add_run(config.digest, protocol(), SeedBundle.create(7, 0))
    assert study.add_run(config.digest, protocol(), SeedBundle.create(7, 0)) == run
    study.update_run(run, "running", 1, 32, "checkpoint")
    observation = study.add_observation(
        config.digest,
        objectives().digest,
        "quality.nll",
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
