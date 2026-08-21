"""whole-partition quality evaluation workers for version three search."""

import torch
import torch.nn.functional as F

from speck.model_v3 import SpeckV3ForCausalLM
from speck.search.artifacts import ArtifactStore
from speck.search.checkpoints import load_run_checkpoint
from speck.search.initialize_v3 import git_state, runtime_environment
from speck.search.protocol import worker_protocol_version
from speck.search.quality_worker import LeaseHeartbeat
from speck.search.segments import load_segment_plan, segment_evaluation_batches
from speck.search.study_v3 import V3Study
from speck.tokenizer import get_tokenizer


def execute_evaluation_action(
    study_path,
    artifact_root,
    action,
    *,
    owner,
    device,
    lease_seconds,
    tokenizer=None,
):
    if action["status"] != "running" or action["owner"] != owner:
        raise ValueError("evaluation action is not owned by this worker")
    payload = action["payload"]
    if payload.get("worker_protocol_version") != worker_protocol_version:
        raise ValueError("evaluation action worker protocol does not match")
    artifacts = ArtifactStore(artifact_root)
    study = V3Study(study_path, readonly=True)
    try:
        stored = study.study()
        tokenizer = tokenizer or get_tokenizer(**stored["provenance"]["tokenizer"])
        run = study.run(action["run_id"])
        architecture = study.architecture(run["architecture_digest"])["config"]
        checkpoint = study.checkpoint(payload["checkpoint_digest"])
        protocol = run["protocol"]
        plan = load_segment_plan(stored["provenance"]["segment_plan"]["path"])
    finally:
        study.close()
    if run["checkpoint_digest"] != checkpoint.artifact.digest:
        raise ValueError("evaluation checkpoint is no longer current")
    if plan.digest != protocol.segment_plan_digest:
        raise ValueError("evaluation segment plan identity changed")
    if tokenizer.fingerprint() != protocol.tokenizer_digest:
        raise ValueError("evaluation tokenizer identity changed")
    device = torch.device(device)
    if device.type != protocol.device_type:
        raise ValueError("evaluation worker device does not match its protocol")
    dtypes = {"bfloat16": torch.bfloat16, "float32": torch.float32}
    heartbeat = LeaseHeartbeat(
        study_path,
        action["id"],
        action["claim_token"],
        lease_seconds,
    )
    heartbeat.start()
    try:
        model = SpeckV3ForCausalLM(architecture).to(
            device=device,
            dtype=dtypes[protocol.dtype],
        )
        checkpoint_payload = load_run_checkpoint(
            artifacts,
            checkpoint,
            device=device,
            architecture_digest=architecture.digest,
            protocol_digest=protocol.digest,
            seed_bundle_digest=run["seed_bundle_digest"],
        )
        model.load_state_dict(checkpoint_payload["model"])
        model.eval()
        loss_sum = 0.0
        evaluated_tokens = 0
        with torch.inference_mode():
            for inputs, targets in segment_evaluation_batches(
                tokenizer,
                plan,
                protocol.evaluation_partition,
                protocol.evaluation_batch_size,
                protocol.sequence_length,
                device=device,
                data_dir=stored["provenance"]["dataset_dir"],
            ):
                logits = model(inputs)
                loss = F.cross_entropy(
                    logits.flatten(0, 1),
                    targets.flatten(),
                    reduction="sum",
                )
                loss_sum += float(loss)
                evaluated_tokens += targets.numel()
                heartbeat.check()
        nll = loss_sum / evaluated_tokens
        artifact = artifacts.put_json(
            "quality_evaluation",
            {
                "architecture_digest": architecture.digest,
                "checkpoint_digest": checkpoint.artifact.digest,
                "environment": runtime_environment(),
                "evaluated_tokens": evaluated_tokens,
                "loss_sum": loss_sum,
                "nll": nll,
                "partition": protocol.evaluation_partition,
                "protocol_digest": protocol.digest,
                "run_id": action["run_id"],
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
        observation_ids = study.commit_quality_evaluation(
            action["id"],
            action["claim_token"],
            nll,
            evaluated_tokens,
            artifact,
        )
    finally:
        study.close()
    return {
        "action_id": action["id"],
        "artifact_digest": artifact.digest,
        "evaluated_tokens": evaluated_tokens,
        "nll": nll,
        "observation_ids": observation_ids,
    }


def run_evaluation_worker(
    study_path,
    artifact_root,
    *,
    owner,
    device,
    lease_seconds=300,
    tokenizer=None,
    captured_git=None,
):
    device_type = torch.device(device).type
    study = V3Study(study_path)
    try:
        expected_git = study.study()["provenance"]["git"]
        if (captured_git or git_state()) != expected_git:
            raise RuntimeError("study code changed before quality evaluation")
        study.release_expired_actions()
        action = study.claim_action(
            owner,
            lease_seconds=lease_seconds,
            kind="evaluate",
            device_type=device_type,
        )
    finally:
        study.close()
    if action is None:
        return None
    try:
        return execute_evaluation_action(
            study_path,
            artifact_root,
            action,
            owner=owner,
            device=device,
            lease_seconds=lease_seconds,
            tokenizer=tokenizer,
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
