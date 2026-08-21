"""isolated leased profiling workers for version three search."""

import torch

from speck.profile.backends.torch_native import TorchNativeBackend
from speck.profile.protocol import profile_session
from speck.profile.schema import ProfileScenario
from speck.search.artifacts import ArtifactStore
from speck.search.initialize_v3 import git_state, runtime_environment
from speck.search.protocol import worker_protocol_version
from speck.search.quality_worker import LeaseHeartbeat
from speck.search.study_v3 import V3Study


def backend_plugin(name):
    if name == "torch_native":
        return TorchNativeBackend()
    raise ValueError(f"unknown profile backend: {name}")


def execute_profile_action(
    study_path,
    artifact_root,
    action,
    *,
    owner,
    device,
    lease_seconds,
):
    if action["status"] != "running" or action["owner"] != owner:
        raise ValueError("profile action is not owned by this worker")
    payload = action["payload"]
    if payload.get("worker_protocol_version") != worker_protocol_version:
        raise ValueError("profile action worker protocol does not match")
    scenario = ProfileScenario.from_dict(payload["scenario"])
    if scenario.digest != payload["scenario_digest"]:
        raise ValueError("profile scenario digest changed")
    if torch.device(device).type != scenario.device.split(":", 1)[0]:
        raise ValueError("profile worker device does not match its scenario")
    backend = backend_plugin(scenario.backend.name)
    if backend.identity != scenario.backend:
        raise ValueError("profile backend identity changed")
    study = V3Study(study_path, readonly=True)
    try:
        architecture = study.architecture(payload["architecture_digest"])["config"]
    finally:
        study.close()
    supported, reason = backend.supports(architecture, scenario)
    if not supported:
        raise ValueError(reason)

    heartbeat = LeaseHeartbeat(
        study_path,
        action["id"],
        action["claim_token"],
        lease_seconds,
    )
    heartbeat.start()
    try:
        torch.manual_seed(payload["prompt_seed"])
        prepared = backend.prepare(architecture, scenario)
        session = backend.load(prepared, scenario)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(payload["prompt_seed"])
        prompt = torch.randint(
            0,
            architecture.vocab_size,
            (scenario.batch_size, scenario.prompt_tokens),
            generator=generator,
        )
        generated = torch.randint(
            0,
            architecture.vocab_size,
            (scenario.batch_size, scenario.generated_tokens),
            generator=generator,
        )
        try:
            result = profile_session(
                session,
                scenario,
                architecture.digest,
                prompt,
                generated,
                prepared.weight_bytes,
            )
        finally:
            session.close()
        artifacts = ArtifactStore(artifact_root)
        artifact = artifacts.put_json(
            "profile_result",
            {
                "environment": runtime_environment(),
                "repetition": payload["repetition"],
                "result": result.export(),
                "scenario": scenario.export(),
            },
        )
    finally:
        heartbeat.stop()
    study = V3Study(study_path)
    try:
        study.heartbeat_action(
            action["id"],
            action["claim_token"],
            lease_seconds,
        )
        observation_ids = study.commit_profile_result(
            action["id"],
            action["claim_token"],
            scenario,
            result,
            artifact,
        )
    finally:
        study.close()
    return {
        "action_id": action["id"],
        "artifact_digest": artifact.digest,
        "observation_ids": observation_ids,
        "result_digest": result.digest,
    }


def run_profile_worker(
    study_path,
    artifact_root,
    *,
    owner,
    device,
    backend="torch_native",
    lease_seconds=300,
    captured_git=None,
):
    plugin = backend_plugin(backend)
    device_type = torch.device(device).type
    study = V3Study(study_path)
    try:
        expected_git = study.study()["provenance"]["git"]
        if (captured_git or git_state()) != expected_git:
            raise RuntimeError("study code changed before profiling")
        study.release_expired_actions()
        action = study.claim_action(
            owner,
            lease_seconds=lease_seconds,
            kind="profile",
            backend=plugin.identity.name,
            device_type=device_type,
        )
    finally:
        study.close()
    if action is None:
        return None
    try:
        return execute_profile_action(
            study_path,
            artifact_root,
            action,
            owner=owner,
            device=device,
            lease_seconds=lease_seconds,
        )
    except Exception as error:
        study = V3Study(study_path)
        try:
            try:
                study.finish_action(
                    action["id"],
                    action["claim_token"],
                    error=f"{type(error).__name__}: {error}",
                )
            except RuntimeError:
                pass
        finally:
            study.close()
        raise
