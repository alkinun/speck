"""Run distributed Speck pretraining with validation, checkpoints, and W&B logging."""

import argparse
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel

from speck.architecture import ArchitectureConfig
from speck.checkpoint import (
    checkpoint_identity,
    latest,
    load,
    load_metadata,
    load_timing,
    save,
)
from speck.common import NullRun, base_dir, cleanup, init_runtime, print0
from speck.config import load_experiment
from speck.dataloader import manifest_fingerprint, packed_loader
from speck.dataset import load_manifest, resolve_data_dir, verify_shards
from speck.model import build_model
from speck.tokenizer import get_tokenizer
from speck.train import (
    average_loss,
    branch_position,
    checkpoint_global_tokens,
    checkpoint_milestones,
    lr_scale,
    optimization_step,
    resolve_device_batch_size,
    validate_loader_progress,
)

_BRANCH_FIXED_SETTINGS = (
    "sequence_length",
    "device_batch_size",
    "batch_tokens",
    "weight_decay",
    "grad_clip",
    "optimizer",
    "world_size",
    "seed",
)
_SCHEDULE_SETTINGS = ("lr", "warmup_steps", "min_lr", "lr_schedule")
_CONTEXT_FIXED_SETTINGS = ("weight_decay", "grad_clip", "optimizer", "seed")
_IMMUTABLE_RESUME_SETTINGS = (
    "sequence_length",
    "device_batch_size",
    "batch_tokens",
    "train_tokens",
    "lr",
    "weight_decay",
    "warmup_steps",
    "min_lr",
    "lr_schedule",
    "grad_clip",
    "optimizer",
    "world_size",
    "global_token_offset",
    "checkpoint_tokens",
    "training_phase",
    "branch_kind",
    "seed",
)
_LEGACY_RESUME_DEFAULTS = {
    "lr_schedule": "cosine",
    "global_token_offset": 0,
    "checkpoint_tokens": [],
    "training_phase": "base",
    "seed": 42,
    "branch_kind": "same",
}


def changed_resume_settings(previous, current):
    return [
        key
        for key in _IMMUTABLE_RESUME_SETTINGS
        if previous.get(key, _LEGACY_RESUME_DEFAULTS.get(key))
        != current.get(key, _LEGACY_RESUME_DEFAULTS.get(key))
    ]


def changed_branch_settings(previous, current, allow_schedule_change=False):
    settings = _BRANCH_FIXED_SETTINGS
    if not allow_schedule_change:
        settings += _SCHEDULE_SETTINGS
    return [
        key
        for key in settings
        if previous.get(key, _LEGACY_RESUME_DEFAULTS.get(key))
        != current.get(key, _LEGACY_RESUME_DEFAULTS.get(key))
    ]


def changed_context_settings(previous, current):
    """Return optimizer semantics that a context-extension branch tried to change."""

    return [
        key
        for key in _CONTEXT_FIXED_SETTINGS
        if previous.get(key, _LEGACY_RESUME_DEFAULTS.get(key))
        != current.get(key, _LEGACY_RESUME_DEFAULTS.get(key))
    ]


def context_compatible_architecture(previous, current):
    """Allow positional capacity changes without allowing parameter-topology drift."""

    ignored = {"max_position_embeddings", "rope_theta", "rope_scaling_factor"}
    previous = ArchitectureConfig.from_dict(previous).settings()
    current = ArchitectureConfig.from_dict(current).settings()
    return {key: value for key, value in previous.items() if key not in ignored} == {
        key: value for key, value in current.items() if key not in ignored
    }


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment",
        nargs="?",
        default="experiments/Speck1-140M",
        help="experiment directory (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="training device; defaults to automatic runtime selection",
    )
    parser.add_argument(
        "--resume",
        type=int,
        default=None,
        help="checkpoint step to resume from",
    )
    parser.add_argument(
        "--branch-from",
        type=Path,
        default=None,
        help="complete parent checkpoint directory for a new same-recipe branch",
    )
    parser.add_argument(
        "--branch-step",
        type=int,
        default=None,
        help="parent checkpoint step used with --branch-from",
    )
    parser.add_argument(
        "--branch-schedule",
        choices=("inherit", "new"),
        default="inherit",
        help="inherit the parent schedule or start the branch schedule at step zero",
    )
    parser.add_argument(
        "--branch-kind",
        choices=("same", "context"),
        default="same",
        help="same-recipe comparison or explicit progressive-context continuation",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="disable torch.compile",
    )
    parser.add_argument(
        "--device-batch-size",
        type=int,
        default=None,
        help="per-device batch ceiling; defaults to the experiment configuration",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="runtime checkpoint interval in steps; zero disables periodic saves",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help="runtime validation interval in steps; zero disables periodic evaluation",
    )
    parser.add_argument(
        "--stop-at-tokens",
        type=int,
        default=None,
        help="stop at a configured token milestone and write a resumable checkpoint",
    )
    return parser.parse_args(argv)


@torch.no_grad()
def validate(model, loader, steps, world_size, source_ids):
    model.eval()
    device = next(model.parameters()).device
    source_indices = {source_id: index for index, source_id in enumerate(source_ids)}
    losses = torch.zeros(len(source_ids), device=device)
    counts = torch.zeros(len(source_ids), device=device)
    for _ in range(steps):
        inputs, targets, state = next(loader)
        index = source_indices[state["selected_source"]]
        losses[index] += model(inputs, targets)
        counts[index] += 1
    if world_size > 1:
        dist.all_reduce(losses)
        dist.all_reduce(counts)
    model.train()
    source_losses = {
        source_id: (losses[index] / counts[index]).item()
        for index, source_id in enumerate(source_ids)
        if counts[index].item()
    }
    return (losses.sum() / counts.sum()).item(), source_losses


class BaseTrainer:
    """Own one base-training lifecycle and its mutable execution state."""

    def __init__(self, configs, cli):
        self.configs = configs
        self.cli = cli
        self.session_started = time.perf_counter()
        self.tracking: Any = None
        self._prepare_settings()
        self._load_checkpoint_metadata()

    def _prepare_settings(self):
        args = SimpleNamespace(**self.configs["train"])
        args.run = args.run or Path(self.cli.experiment).resolve().name
        args.device = self.cli.device
        args.resume = self.cli.resume
        args.no_compile = self.cli.no_compile
        args.global_token_offset = getattr(args, "global_token_offset", 0)
        args.checkpoint_tokens = getattr(args, "checkpoint_tokens", [])
        args.training_phase = getattr(args, "training_phase", "base")
        args.branch_kind = self.cli.branch_kind
        args.lr_schedule = getattr(args, "lr_schedule", "cosine")
        args.wandb_group = getattr(args, "wandb_group", None)
        args.seed = getattr(args, "seed", 42)
        args.stop_at_tokens = getattr(self.cli, "stop_at_tokens", None)
        if not isinstance(args.seed, int) or isinstance(args.seed, bool):
            raise ValueError("seed must be an integer")
        if args.stop_at_tokens is not None and args.stop_at_tokens not in args.checkpoint_tokens:
            raise ValueError("--stop-at-tokens must name a configured checkpoint token milestone")
        for key in ("save_every", "eval_every"):
            override = getattr(self.cli, key, None)
            if override is not None:
                setattr(args, key, override)
            if not isinstance(getattr(args, key), int) or getattr(args, key) < 0:
                raise ValueError(f"{key} must be a non-negative integer")
        args.data_dir = str(
            resolve_data_dir(
                self.configs["data"].get("output_dir"),
                self.configs["data"].get("output_name"),
            )
        )
        args.output_dir = args.output_dir or os.path.join(base_dir(), "checkpoints", args.run)
        self.args = args
        self.branching = self.cli.branch_from is not None or self.cli.branch_step is not None
        if (self.cli.branch_from is None) != (self.cli.branch_step is None):
            raise ValueError("--branch-from and --branch-step must be provided together")
        if args.resume is not None and self.branching:
            raise ValueError("--resume and --branch-from are mutually exclusive")
        if not self.branching and self.cli.branch_schedule != "inherit":
            raise ValueError("--branch-schedule new requires --branch-from")
        if not self.branching and self.cli.branch_kind != "same":
            raise ValueError("--branch-kind context requires --branch-from")
        if self.branching and self.cli.branch_kind == "context":
            if self.cli.branch_schedule != "new":
                raise ValueError("context branches require --branch-schedule new")
            if args.training_phase != "context_extension":
                raise ValueError("context branches require training_phase context_extension")
        if args.resume is None and latest(args.output_dir) is not None:
            raise FileExistsError(
                f"checkpoints already exist: {args.output_dir}; pass --resume STEP"
            )

    def _load_checkpoint_metadata(self):
        args = self.args
        self.metadata = (
            load_metadata(args.output_dir, args.resume) if args.resume is not None else None
        )
        self.parent_directory = (
            self.cli.branch_from.expanduser().resolve() if self.branching else None
        )
        if self.parent_directory == Path(args.output_dir).expanduser().resolve():
            raise ValueError("branch output directory must differ from its parent")
        self.parent_metadata = (
            load_metadata(self.parent_directory, self.cli.branch_step) if self.branching else None
        )
        if self.metadata:
            args.global_token_offset = self.metadata["resolved"].get("global_token_offset", 0)
            args.branch_kind = self.metadata["resolved"].get("branch_kind", "same")
        self.data_token_offset = (
            self.metadata["resolved"].get("data_token_offset", 0) if self.metadata else 0
        )
        if self.parent_metadata and self.cli.branch_kind == "context":
            self.data_token_offset = 0

    def _initialize_runtime(self):
        self.rank, self.local_rank, self.world_size, self.device = init_runtime(self.args.device)
        self.distributed = self.world_size > 1
        self.master = self.rank == 0
        self.parent = self.metadata["resolved"].get("parent_checkpoint") if self.metadata else None
        self.branch_schedule = (
            self.metadata["resolved"].get("branch_schedule") if self.metadata else None
        )
        if self.parent_metadata:
            objects = [
                checkpoint_identity(self.parent_directory, self.cli.branch_step)
                if self.master
                else None
            ]
            if self.distributed:
                dist.broadcast_object_list(objects, src=0)
            self.parent = objects[0]
            self.branch_schedule = self.cli.branch_schedule

    def _load_and_verify_data(self):
        self.tokenizer = get_tokenizer(**self.configs["tokenizer"])
        self.manifest = load_manifest(self.args.data_dir)
        self.manifest_hash = manifest_fingerprint(self.manifest)
        self.source_ids = tuple(source["id"] for source in self.manifest["sources"])
        if self.manifest["tokenizer"]["fingerprint"] != self.tokenizer.fingerprint():
            raise ValueError("dataset and tokenizer do not match")
        error: list[str | None] = [None]
        if self.master:
            try:
                verify_shards(self.args.data_dir, self.manifest)
            except Exception as exception:
                error[0] = str(exception)
        if self.distributed:
            dist.broadcast_object_list(error, src=0)
        if error[0]:
            raise ValueError(error[0])

    def _initialize_model_and_geometry(self):
        args = self.args
        torch.manual_seed(args.seed)
        self.model = build_model(
            self.configs["model"],
            self.tokenizer.vocab_size,
            self.tokenizer.bos_id,
            self.tokenizer.eos_id,
        ).to(self.device)
        self.config = self.model.config
        self.model.init_weights()
        self.parameters = tuple(self.model.parameters())
        self.optimizer = self.model.optimizer(args.lr, args.weight_decay, args.optimizer)
        batch_limit = (
            args.device_batch_size
            if self.cli.device_batch_size is None
            else self.cli.device_batch_size
        )
        args.device_batch_size = resolve_device_batch_size(
            batch_limit,
            args.batch_tokens,
            args.sequence_length,
            self.world_size,
        )
        micro_tokens = args.device_batch_size * args.sequence_length * self.world_size
        self.accumulation = args.batch_tokens // micro_tokens
        self.steps = math.ceil(args.train_tokens / args.batch_tokens)
        self.consumed_tokens = self.steps * args.batch_tokens
        self.schedule_step_offset = 0
        self.schedule_steps = self.steps
        if self.metadata:
            self.schedule_step_offset = self.metadata["resolved"].get("schedule_step_offset", 0)
            self.schedule_steps = self.metadata["resolved"].get("schedule_steps", self.steps)
        elif self.parent_metadata:
            if self.cli.branch_kind == "context":
                args.global_token_offset = checkpoint_global_tokens(
                    self.parent_metadata, args.batch_tokens
                )
                self.data_token_offset = 0
                self.schedule_step_offset = 0
                self.schedule_steps = self.steps
            else:
                (
                    args.global_token_offset,
                    self.data_token_offset,
                    self.schedule_step_offset,
                    self.schedule_steps,
                ) = branch_position(
                    self.parent_metadata,
                    args.batch_tokens,
                    self.steps if self.cli.branch_schedule == "inherit" else None,
                )
                if self.cli.branch_schedule == "new":
                    self.schedule_step_offset = 0
                    self.schedule_steps = self.steps
        if (
            not isinstance(args.global_token_offset, int)
            or isinstance(args.global_token_offset, bool)
            or args.global_token_offset < 0
            or args.global_token_offset % args.batch_tokens
        ):
            raise ValueError("global token offset must align with optimizer batches")
        if self.parent_metadata and self.cli.branch_kind == "context":
            self.global_step_offset = self.parent_metadata.get(
                "global_step", self.parent_metadata["step"]
            )
        else:
            self.global_step_offset = args.global_token_offset // args.batch_tokens
        self.global_consumed_tokens = args.global_token_offset + self.consumed_tokens
        self.milestones = checkpoint_milestones(
            args.checkpoint_tokens,
            args.batch_tokens,
            args.global_token_offset,
            self.steps,
        )
        self.stop_step = None
        if args.stop_at_tokens is not None:
            matches = [
                step for step, tokens in self.milestones.items() if tokens == args.stop_at_tokens
            ]
            if len(matches) != 1:
                raise ValueError("stop token milestone is outside this training phase")
            self.stop_step = matches[0]
        if self.manifest["splits"]["train"]["tokens"] <= self.consumed_tokens:
            raise ValueError("packed dataset is too small for this run")

    def _restore_training_state(self):
        args = self.args
        self.data_state = None
        self.start_step = 0
        self.completed_step = 0
        self.elapsed_training = 0.0
        self.elapsed_optimizer = 0.0
        self.elapsed_evaluation = 0.0
        self.elapsed_active = 0.0
        self.elapsed_checkpoint = 0.0
        if args.resume is not None:
            metadata = self.metadata
            if metadata is None:
                raise RuntimeError("resume metadata was not loaded")
            model_state, optimizer_state, loaded_metadata = load(
                args.output_dir, args.resume, self.device
            )
            if loaded_metadata != metadata:
                raise ValueError("checkpoint metadata changed while loading")
            stored_config = ArchitectureConfig.from_dict(metadata["config"]).settings()
            if (
                stored_config != self.config.settings()
                or metadata["manifest"] != self.manifest_hash
            ):
                raise ValueError("checkpoint does not match the model or dataset")
            self.model.load_state_dict(model_state)
            self.optimizer.load_state_dict(optimizer_state)
            self.start_step = metadata["step"]
            self.completed_step = self.start_step
            self.data_state = metadata["data_state"]
            validate_loader_progress(
                self.data_state,
                self.data_token_offset + self.start_step * args.batch_tokens,
            )
            self.elapsed_training = metadata["training_seconds"]
            timing = load_timing(args.output_dir, args.resume) or metadata.get("timing", {})
            self.elapsed_optimizer = timing.get("optimizer_seconds", self.elapsed_training)
            self.elapsed_evaluation = timing.get("evaluation_seconds", 0.0)
            self.elapsed_active = timing.get("active_seconds", self.elapsed_training)
            self.elapsed_checkpoint = timing.get("checkpoint_seconds", 0.0)
        elif self.parent_metadata:
            self._restore_parent_checkpoint()

    def _restore_parent_checkpoint(self):
        parent_metadata = self.parent_metadata
        if parent_metadata is None:
            raise RuntimeError("branch metadata was not loaded")
        context_branch = self.cli.branch_kind == "context"
        stored_config = ArchitectureConfig.from_dict(parent_metadata["config"]).settings()
        architecture_matches = (
            context_compatible_architecture(parent_metadata["config"], self.config.export())
            if context_branch
            else stored_config == self.config.settings()
        )
        manifest_matches = context_branch or parent_metadata["manifest"] == self.manifest_hash
        if not architecture_matches or not manifest_matches:
            raise ValueError("branch parent does not match the model or dataset")
        branch_settings = {**vars(self.args), "world_size": self.world_size}
        changed = (
            changed_context_settings(parent_metadata["resolved"], branch_settings)
            if context_branch
            else changed_branch_settings(
                parent_metadata["resolved"],
                branch_settings,
                allow_schedule_change=self.cli.branch_schedule == "new",
            )
        )
        if changed:
            raise ValueError(f"branch settings changed: {', '.join(changed)}")
        model_state, optimizer_state, loaded_parent = load(
            self.parent_directory, self.cli.branch_step, self.device
        )
        if loaded_parent != parent_metadata:
            raise ValueError("parent checkpoint metadata changed while loading")
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)
        if context_branch:
            self.data_state = None
        else:
            self.data_state = parent_metadata["data_state"]
            validate_loader_progress(self.data_state, self.data_token_offset)

    def _build_resolved_settings(self):
        dataset_provenance = {
            "format": self.manifest["format"],
            "requested_train_tokens": self.manifest["requested_train_tokens"],
            "mixture": self.manifest["mixture"],
            "sources": [
                {
                    "id": source["id"],
                    "repo": source["repo"],
                    "revision": source["revision"],
                    "file_list_sha256": source["file_list_sha256"],
                }
                for source in self.manifest["sources"]
            ],
        }
        self.resolved = {
            **vars(self.args),
            "experiment": str(Path(self.cli.experiment).resolve()),
            "tokenizer": self.configs["tokenizer"],
            "model": self.config.export(),
            "parameters": self.model.parameter_count(),
            "optimizer_roles": self.model.optimizer_role_counts(self.optimizer),
            "manifest": self.manifest_hash,
            "dataset": dataset_provenance,
            "world_size": self.world_size,
            "accumulation_steps": self.accumulation,
            "steps": self.steps,
            "consumed_tokens": self.consumed_tokens,
            "global_step_offset": self.global_step_offset,
            "global_consumed_tokens": self.global_consumed_tokens,
            "data_token_offset": self.data_token_offset,
            "schedule_step_offset": self.schedule_step_offset,
            "schedule_steps": self.schedule_steps,
            "parent_checkpoint": self.parent,
            "branch_schedule": self.branch_schedule,
            "branch_kind": self.args.branch_kind,
            "milestone_steps": {str(step): token for step, token in self.milestones.items()},
        }
        if self.metadata:
            changed = changed_resume_settings(self.metadata["resolved"], self.resolved)
            if changed:
                raise ValueError(f"resume settings changed: {', '.join(changed)}")
        print0(json.dumps(self.resolved, indent=2, sort_keys=True))

    def _initialize_tracking(self):
        args = self.args
        if self.master and args.run != "dummy":
            self.tracking = wandb.init(
                project=args.wandb_project,
                name=args.run,
                group=args.wandb_group,
                job_type=args.training_phase,
                id=self.metadata.get("wandb_id") if self.metadata else None,
                resume="must" if self.metadata and self.metadata.get("wandb_id") else None,
                config=self.resolved,
            )
            wandb.define_metric("progress/step")
            wandb.define_metric("*", step_metric="progress/step")
        else:
            self.tracking = NullRun()

    def _prepare_execution(self):
        args = self.args
        self.train_data = packed_loader(
            self.tokenizer,
            args.device_batch_size,
            args.sequence_length,
            "train",
            device=self.device,
            resume_state_dict=self.data_state,
            data_dir=args.data_dir,
        )
        self.inputs, self.targets, self.data_state = next(self.train_data)
        train_model: Any = self.model
        if self.distributed:
            train_model = DistributedDataParallel(
                train_model,
                device_ids=[self.local_rank],
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )
        self.train_model: Any = (
            train_model
            if args.no_compile
            else torch.compile(
                train_model,
                dynamic=False,
                options={
                    "max_autotune": True,
                    "coordinate_descent_tuning": True,
                    "aggressive_fusion": True,
                },
            )
        )
        compile_step = getattr(self.optimizer, "compile_step", None)
        if not args.no_compile and compile_step is not None:
            compile_step()
        self.flops = self.model.flops_per_token(args.sequence_length)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def _validate(self, step):
        started = time.perf_counter()
        args = self.args
        tokens_per_step = args.device_batch_size * args.sequence_length * self.world_size
        eval_tokens = args.final_eval_tokens if step == self.steps else args.eval_tokens
        val_steps = max(
            1,
            min(eval_tokens, self.manifest["splits"]["val"]["tokens"]) // tokens_per_step,
        )
        loader = packed_loader(
            self.tokenizer,
            args.device_batch_size,
            args.sequence_length,
            "val",
            device=self.device,
            data_dir=args.data_dir,
        )
        loss, source_losses = validate(
            self.train_model,
            loader,
            val_steps,
            self.world_size,
            self.source_ids,
        )
        evaluated_tokens = val_steps * tokens_per_step
        self.elapsed_evaluation += time.perf_counter() - started
        global_step = self.global_step_offset + step
        global_tokens = args.global_token_offset + step * args.batch_tokens
        metrics = {
            "progress/step": global_step,
            "progress/phase_step": step,
            "progress/tokens": global_tokens,
            "validation/loss": loss,
            "validation/perplexity": math.exp(min(loss, 20)),
            "validation/tokens": evaluated_tokens,
        }
        for source_id, source_loss in source_losses.items():
            metrics[f"validation/source/{source_id}/loss"] = source_loss
            metrics[f"validation/source/{source_id}/perplexity"] = math.exp(min(source_loss, 20))
        self.tracking.log(metrics)
        print0(f"step {global_step:,} ({global_tokens:,} tokens) | validation loss {loss:.5f}")
        return loss, source_losses, evaluated_tokens

    def _checkpoint(
        self,
        step,
        validation_loss,
        validation_source_losses,
        validation_step,
        validation_tokens,
        milestone,
    ):
        started = time.perf_counter()
        args = self.args
        if self.master:
            global_tokens = args.global_token_offset + step * args.batch_tokens
            state = {
                "step": step,
                "global_step": self.global_step_offset + step,
                "global_tokens": global_tokens,
                "training_phase": args.training_phase,
                "config": self.config.settings(),
                "resolved": self.resolved,
                "manifest": self.manifest_hash,
                "data_state": self.data_state,
                "validation_loss": validation_loss,
                "validation_source_losses": validation_source_losses,
                "validation_step": validation_step,
                "validation_global_tokens": (
                    args.global_token_offset + validation_step * args.batch_tokens
                    if validation_step is not None
                    else None
                ),
                "validation_tokens": validation_tokens,
                "milestone_tokens": milestone,
                "partial": step < self.steps,
                "training_seconds": self.elapsed_training,
                "timing": {
                    "optimizer_seconds": self.elapsed_optimizer,
                    "evaluation_seconds": self.elapsed_evaluation,
                    "checkpoint_seconds": self.elapsed_checkpoint,
                    "active_seconds": self.elapsed_active
                    + time.perf_counter()
                    - self.session_started,
                },
                "wandb_id": self.tracking.id,
            }
            save(
                args.output_dir,
                step,
                self.model.state_dict(),
                self.optimizer.state_dict(),
                state,
                timing=lambda: {
                    "optimizer_seconds": self.elapsed_optimizer,
                    "evaluation_seconds": self.elapsed_evaluation,
                    "checkpoint_seconds": self.elapsed_checkpoint + time.perf_counter() - started,
                    "active_seconds": self.elapsed_active
                    + time.perf_counter()
                    - self.session_started,
                },
            )
        if self.distributed:
            dist.barrier()
        self.elapsed_checkpoint += time.perf_counter() - started

    def _initial_validation(self):
        metadata = self.metadata
        if metadata:
            return (
                metadata["validation_loss"],
                metadata.get("validation_source_losses", {}),
                metadata.get("validation_step"),
                metadata.get("validation_tokens", 0),
            )
        loss, source_losses, tokens = self._validate(0)
        return loss, source_losses, 0, tokens

    def _run_steps(self):
        args = self.args
        validation_loss, validation_source_losses, validation_step, validation_tokens = (
            self._initial_validation()
        )
        synchronize = torch.cuda.synchronize if self.device.type == "cuda" else lambda: None
        timing_started = time.perf_counter()
        timing_steps = 0
        for step in range(self.start_step, self.steps):
            completed = step + 1
            should_log = completed == 1 or completed % args.log_every == 0
            milestone = self.milestones.get(completed)
            stop_now = self.stop_step == completed
            should_validate = (
                (args.eval_every > 0 and completed % args.eval_every == 0)
                or milestone is not None
                or completed == self.steps
            )
            should_save = (
                (args.save_every > 0 and completed % args.save_every == 0)
                or milestone is not None
                or completed == self.steps
            )
            scale = lr_scale(
                self.schedule_step_offset + step,
                self.schedule_steps,
                args.warmup_steps,
                args.min_lr,
                args.lr_schedule,
            )
            loss, grad_norm, batch = optimization_step(
                self.train_model,
                self.parameters,
                self.optimizer,
                self.train_data,
                (self.inputs, self.targets, self.data_state),
                self.accumulation,
                args.grad_clip,
                args.lr * scale,
                self.distributed,
            )
            self.inputs, self.targets, self.data_state = batch
            self.completed_step = completed
            timing_steps += 1
            should_flush_timing = should_log or should_validate or should_save or completed == 10
            duration = None
            if should_flush_timing:
                synchronize()
                window_duration = time.perf_counter() - timing_started
                self.elapsed_optimizer += window_duration
                if completed > 10:
                    self.elapsed_training += window_duration
                duration = window_duration / timing_steps
            if self.distributed and should_log:
                average_loss(loss, True)
            if should_log:
                self._log_step(completed, loss, grad_norm, duration)
            if should_validate:
                validation_loss, validation_source_losses, validation_tokens = self._validate(
                    completed
                )
                validation_step = completed
            if should_save:
                self._checkpoint(
                    completed,
                    validation_loss,
                    validation_source_losses,
                    validation_step,
                    validation_tokens,
                    milestone,
                )
            if should_flush_timing:
                timing_started = time.perf_counter()
                timing_steps = 0
            if stop_now:
                break

    def _log_step(self, completed, loss, grad_norm, duration):
        assert duration is not None
        args = self.args
        data_state = self.data_state
        if data_state is None:
            raise RuntimeError("training data state is unavailable")
        metrics = {
            "progress/step": self.global_step_offset + completed,
            "progress/phase_step": completed,
            "progress/tokens": args.global_token_offset + completed * args.batch_tokens,
            "train/loss": loss.item(),
            "train/lr": float(self.optimizer.param_groups[0]["lr"]),
            "train/grad_norm": float(grad_norm),
            "performance/tokens_per_second": args.batch_tokens / duration,
            "performance/tflops": self.flops * args.batch_tokens / duration / 1e12,
            "model/parameters": self.model.parameter_count(),
            "data/next_source": data_state["selected_source"],
            "data/next_source_epoch": data_state["source_epochs"][data_state["selected_source"]],
            "data/next_phase": data_state["phase"],
            "data/next_shard": data_state["shard"]["index"],
        }
        active_seconds = self.elapsed_active + time.perf_counter() - self.session_started
        metrics["performance/gpu_hours"] = active_seconds * self.world_size / 3600
        if self.device.type == "cuda":
            metrics["performance/peak_allocated_vram_mib"] = (
                torch.cuda.max_memory_allocated(self.device) / 2**20
            )
            metrics["performance/peak_reserved_vram_mib"] = (
                torch.cuda.max_memory_reserved(self.device) / 2**20
            )
        self.tracking.log(metrics)
        print0(
            f"step {metrics['progress/step']:,}/{self.global_step_offset + self.steps:,} | "
            f"loss {metrics['train/loss']:.5f} | "
            f"{metrics['performance/tokens_per_second']:,.0f} tok/s"
        )

    def _write_summary(self):
        if not self.master:
            return
        summary = {
            "training_phase": self.args.training_phase,
            "steps": self.steps,
            "completed_steps": self.completed_step,
            "global_step": self.global_step_offset + self.completed_step,
            "global_tokens": (
                self.args.global_token_offset + self.completed_step * self.args.batch_tokens
            ),
            "partial": self.completed_step < self.steps,
            "stop_at_tokens": self.args.stop_at_tokens,
            "optimizer_seconds": self.elapsed_optimizer,
            "evaluation_seconds": self.elapsed_evaluation,
            "checkpoint_seconds": self.elapsed_checkpoint,
            "active_seconds": self.elapsed_active + time.perf_counter() - self.session_started,
        }
        path = Path(self.args.output_dir) / "run_summary.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _finish_tracking(self):
        if self.tracking is None:
            return
        tracking, self.tracking = self.tracking, None
        tracking.finish()

    def run(self):
        try:
            self._initialize_runtime()
            self._load_and_verify_data()
            self._initialize_model_and_geometry()
            self._restore_training_state()
            self._build_resolved_settings()
            self._initialize_tracking()
            self._prepare_execution()
            self._run_steps()
            self._finish_tracking()
            self._write_summary()
        finally:
            self._finish_tracking()
            cleanup()


def train(configs, cli):
    BaseTrainer(configs, cli).run()


def main():
    cli = arguments()
    configs = load_experiment(cli.experiment, "data", "tokenizer", "model", "train")
    train(configs, cli)


if __name__ == "__main__":
    main()
