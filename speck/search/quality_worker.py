"""leased resumable quality-training workers for version three search."""

import random
import threading
from pathlib import Path

import numpy as np
import torch

from speck.model_v3 import SpeckV3ForCausalLM
from speck.search.artifacts import ArtifactStore
from speck.search.checkpoints import (
    load_run_checkpoint,
    restore_rng_state,
    save_run_checkpoint,
)
from speck.search.initialize_v3 import git_state
from speck.search.protocol import worker_protocol_version
from speck.search.segments import load_segment_plan, segment_loader
from speck.search.study_v3 import V3Study
from speck.tokenizer import get_tokenizer
from speck.train import lr_scale, optimization_step


def _seed_everything(seed, device_type):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if device_type == "cuda":
        torch.cuda.manual_seed_all(seed)


class LeaseHeartbeat:
    def __init__(self, study_path, action_id, claim_token, lease_seconds):
        self.study_path = study_path
        self.action_id = action_id
        self.claim_token = claim_token
        self.lease_seconds = lease_seconds
        self.interval = max(0.1, lease_seconds / 3)
        self.stopped = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        study = None
        try:
            study = V3Study(self.study_path)
            while not self.stopped.wait(self.interval):
                study.heartbeat_action(
                    self.action_id,
                    self.claim_token,
                    self.lease_seconds,
                )
        except Exception as error:
            self.error = error
            self.stopped.set()
        finally:
            if study is not None:
                study.close()

    def start(self):
        self.thread.start()

    def check(self):
        if self.error is not None:
            raise RuntimeError("quality worker lost its action lease") from self.error

    def stop(self):
        self.stopped.set()
        self.thread.join()
        self.check()


def _load_identity(study, action, artifacts, tokenizer):
    if action["kind"] != "continue" or action["run_id"] is None:
        raise ValueError("quality workers need a run-linked continue action")
    if action["payload"].get("worker_protocol_version") != worker_protocol_version:
        raise ValueError("quality action worker protocol does not match")
    run = study.run(action["run_id"])
    architecture = study.architecture(run["architecture_digest"])["config"]
    stored = study.study()
    provenance = stored["provenance"]
    protocol = run["protocol"]
    plan = load_segment_plan(provenance["segment_plan"]["path"])
    if plan.digest != protocol.segment_plan_digest:
        raise ValueError("quality run segment plan identity changed")
    artifacts.verify(study.artifact(plan.digest))
    if tokenizer.fingerprint() != protocol.tokenizer_digest:
        raise ValueError("quality run tokenizer identity changed")
    if action["payload"]["target_tokens"] not in protocol.checkpoint_tokens:
        raise ValueError("quality action target is outside its protocol")
    parent = (
        study.checkpoint(run["checkpoint_digest"])
        if run["checkpoint_digest"] is not None
        else None
    )
    return architecture, plan, protocol, provenance, run, parent


def _execute_quality_action(
    study_path,
    artifact_root,
    action,
    *,
    owner,
    device,
    lease_seconds,
    tokenizer=None,
    heartbeat,
):
    if action["status"] != "running" or action["owner"] != owner:
        raise ValueError("quality action is not owned by this worker")
    artifacts = ArtifactStore(artifact_root)
    study = V3Study(study_path)
    try:
        stored = study.study()
        tokenizer = tokenizer or get_tokenizer(**stored["provenance"]["tokenizer"])
        architecture, plan, protocol, provenance, run, parent = _load_identity(
            study,
            action,
            artifacts,
            tokenizer,
        )
    finally:
        study.close()
    device = torch.device(device)
    if device.type != protocol.device_type:
        raise ValueError("quality worker device does not match its protocol")
    if protocol.compile_model:
        raise ValueError("compiled v3 quality resume is not implemented")
    dtypes = {"bfloat16": torch.bfloat16, "float32": torch.float32}
    data_dir = Path(provenance["dataset_dir"])
    resume_state = None

    _seed_everything(run["seed_bundle"].initialization_seed, device.type)
    model = SpeckV3ForCausalLM(architecture).to(
        device=device,
        dtype=dtypes[protocol.dtype],
    )
    model.init_weights()
    parameters = tuple(model.parameters())
    optimizer = model.optimizer(
        protocol.learning_rate,
        protocol.weight_decay,
        protocol.optimizer,
    )
    payload = None
    if parent is not None:
        payload = load_run_checkpoint(
            artifacts,
            parent,
            device=device,
            architecture_digest=architecture.digest,
            protocol_digest=protocol.digest,
            seed_bundle_digest=run["seed_bundle_digest"],
        )
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        resume_state = payload["data_state"]
    loader = segment_loader(
        tokenizer,
        plan,
        "train",
        run["seed_bundle"].data_seed,
        protocol.device_batch_size,
        protocol.sequence_length,
        device=device,
        resume_state_dict=resume_state,
        data_dir=data_dir,
    )
    batch = next(loader)
    if payload is None:
        _seed_everything(run["seed_bundle"].numerical_seed, device.type)
    else:
        restore_rng_state(payload["rng"])
    accumulation = protocol.batch_tokens // (
        protocol.device_batch_size * protocol.sequence_length
    )
    target_steps = action["payload"]["target_tokens"] // protocol.batch_tokens
    total_steps = protocol.target_tokens // protocol.batch_tokens
    step = run["steps"]
    last_loss = None
    try:
        while step < target_steps:
            heartbeat.check()
            scale = lr_scale(
                step,
                total_steps,
                protocol.warmup_steps,
                protocol.minimum_learning_rate_scale,
            )
            loss, _, batch = optimization_step(
                model,
                parameters,
                optimizer,
                loader,
                batch,
                accumulation,
                protocol.gradient_clip,
                protocol.learning_rate * scale,
            )
            last_loss = float(loss)
            step += 1
        checkpoint = save_run_checkpoint(
            artifacts,
            architecture_digest=architecture.digest,
            protocol_digest=protocol.digest,
            seed_bundle_digest=run["seed_bundle_digest"],
            steps=step,
            tokens=step * protocol.batch_tokens,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            data_state=batch[2],
            parent=parent,
            extra={
                "device": str(device),
                "dtype": protocol.dtype,
                "last_loss": last_loss,
                "owner": owner,
                "torch": torch.__version__,
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
        status = study.commit_quality_checkpoint(
            action["id"],
            action["claim_token"],
            checkpoint,
        )
    finally:
        study.close()
    return {
        "action_id": action["id"],
        "checkpoint_digest": checkpoint.artifact.digest,
        "run_id": action["run_id"],
        "status": status,
        "steps": checkpoint.steps,
        "tokens": checkpoint.tokens,
    }


def execute_quality_action(
    study_path,
    artifact_root,
    action,
    *,
    owner,
    device,
    lease_seconds,
    tokenizer=None,
):
    heartbeat = LeaseHeartbeat(
        study_path,
        action["id"],
        action["claim_token"],
        lease_seconds,
    )
    heartbeat.start()
    try:
        return _execute_quality_action(
            study_path,
            artifact_root,
            action,
            owner=owner,
            device=device,
            lease_seconds=lease_seconds,
            tokenizer=tokenizer,
            heartbeat=heartbeat,
        )
    finally:
        if heartbeat.thread.is_alive():
            heartbeat.stop()
        else:
            heartbeat.check()


def run_quality_worker(
    study_path,
    artifact_root,
    *,
    owner,
    device,
    lease_seconds=300,
    tokenizer=None,
    captured_git=None,
):
    study = V3Study(study_path)
    try:
        expected_git = study.study()["provenance"]["git"]
        if (captured_git or git_state()) != expected_git:
            raise RuntimeError("study code changed before quality training")
        study.release_expired_actions()
        action = study.claim_action(
            owner,
            lease_seconds=lease_seconds,
            kind="continue",
        )
    finally:
        study.close()
    if action is None:
        return None
    try:
        return execute_quality_action(
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
