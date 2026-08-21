import json

from speck.architecture import (
    ArchitectureConfig,
    AttentionSpec,
    BlockConfig,
    BlockGroup,
    StageConfig,
)
from speck.profile.backends.torch_native import TorchNativeBackend
from speck.profile.schema import ProfileScenario
from speck.search.artifacts import ArtifactStore
from speck.search.profile_worker import run_profile_worker
from speck.search.protocol import ObjectiveSet, ObjectiveSpec
from speck.search.study_v3 import V3Study


def architecture():
    return ArchitectureConfig(
        (
            BlockGroup(
                BlockConfig(
                    8,
                    (StageConfig((AttentionSpec(4, 1, "sliding", 3),)),),
                )
            ),
        ),
        8,
        vocab_size=16,
        max_position_embeddings=16,
    )


def objectives():
    return ObjectiveSet(
        "cpu_short",
        (
            ObjectiveSpec("quality.target_nll", "minimize", "quality"),
            ObjectiveSpec("cpu_short.prefill_p95", "minimize", "efficiency"),
            ObjectiveSpec("cpu_short.decode_p95", "minimize", "efficiency"),
            ObjectiveSpec("cpu_short.peak_rss", "minimize", "efficiency"),
            ObjectiveSpec("memory.weight_bytes", "minimize", "efficiency"),
            ObjectiveSpec("memory.state_bytes", "minimize", "efficiency"),
        ),
    )


def scenario(name="cpu_short", device="cpu"):
    return ProfileScenario(
        name,
        TorchNativeBackend().identity,
        device,
        "float32",
        "float32",
        1,
        4,
        2,
        0,
        2,
        process_repetitions=2,
    )


def test_profile_worker_filters_capabilities_and_commits_raw_results(tmp_path):
    study_path = tmp_path / "study.sqlite3"
    artifact_root = tmp_path / "artifacts"
    git = {"dirty": False, "revision": "test", "working_tree": "0" * 64}
    study = V3Study(study_path)
    study.initialize_bundle(
        {},
        {"git": git},
        objective_sets=(objectives(),),
        architecture=architecture(),
    )
    cuda_action = study.add_profile_action(
        architecture().digest,
        scenario("gpu_short", "cuda"),
        objectives().digest,
        0,
        1,
        10.0,
        1.0,
    )
    cpu_action = study.add_profile_action(
        architecture().digest,
        scenario(),
        objectives().digest,
        0,
        2,
        1.0,
        1.0,
    )
    assert study.add_profile_action(
        architecture().digest,
        scenario(),
        objectives().digest,
        0,
        2,
        1.0,
        1.0,
    ) == cpu_action
    study.close()

    result = run_profile_worker(
        study_path,
        artifact_root,
        owner="cpu-test",
        device="cpu",
        lease_seconds=30,
        captured_git=git,
    )
    assert result["action_id"] == cpu_action
    study = V3Study(study_path, readonly=True)
    assert study.action(cpu_action)["status"] == "completed"
    assert study.action(cuda_action)["status"] == "pending"
    observations = study.observations(architecture().digest, objectives().digest)
    assert {item["objective_name"] for item in observations} == {
        "cpu_short.decode_p95",
        "cpu_short.peak_rss",
        "cpu_short.prefill_p95",
        "memory.state_bytes",
        "memory.weight_bytes",
    }
    assert all(item["source"] == "profile" for item in observations)
    artifact = study.artifact(result["artifact_digest"])
    study.close()
    payload = json.loads(ArtifactStore(artifact_root).read_bytes(artifact))
    assert len(payload["result"]["model_prefill_ms"]["samples"]) == 2
    assert payload["repetition"] == 0
