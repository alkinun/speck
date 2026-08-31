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
    UpdateMonitor,
    branch_position,
    checkpoint_milestones,
    lr_scale,
    optimization_step,
    resolve_device_batch_size,
    validate_loader_progress,
)

_UPDATE_MONITOR_SUFFIXES = ("q_proj.weight", "input_projection.weight", "gate_proj.weight")
_BRANCH_FIXED_SETTINGS = (
    "sequence_length",
    "device_batch_size",
    "batch_tokens",
    "weight_decay",
    "grad_clip",
    "optimizer",
    "loss_backend",
    "world_size",
)
_SCHEDULE_SETTINGS = ("lr", "warmup_steps", "min_lr", "lr_schedule", "decay_steps")
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
    "decay_steps",
    "grad_clip",
    "optimizer",
    "loss_backend",
    "world_size",
    "global_token_offset",
    "checkpoint_tokens",
    "training_phase",
)
_LEGACY_RESUME_DEFAULTS = {
    "lr_schedule": "cosine",
    "decay_steps": None,
    "loss_backend": "torch",
    "global_token_offset": 0,
    "checkpoint_tokens": [],
    "training_phase": "base",
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


def build_update_monitor(model, optimizer):
    muon = getattr(optimizer, "optimizers", {}).get("muon")
    if muon is None:
        return None
    optimized = {id(parameter) for group in muon.param_groups for parameter in group["params"]}
    named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if id(parameter) in optimized
    ]
    selected = []
    for suffix in _UPDATE_MONITOR_SUFFIXES:
        matches = [item for item in named if item[0].endswith(suffix)]
        if matches:
            selected.append(matches[len(matches) // 2])
    return UpdateMonitor(selected) if selected else None


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


def train(configs, cli):
    session_started = time.perf_counter()
    args = SimpleNamespace(**configs["train"])
    args.device = cli.device
    args.resume = cli.resume
    args.no_compile = cli.no_compile
    args.global_token_offset = getattr(args, "global_token_offset", 0)
    args.checkpoint_tokens = getattr(args, "checkpoint_tokens", [])
    args.training_phase = getattr(args, "training_phase", "base")
    args.loss_backend = getattr(args, "loss_backend", "torch")
    args.lr_schedule = getattr(args, "lr_schedule", "cosine")
    args.decay_steps = getattr(args, "decay_steps", None)
    args.wandb_group = getattr(args, "wandb_group", None)
    for key in ("save_every", "eval_every"):
        override = getattr(cli, key, None)
        if override is not None:
            setattr(args, key, override)
        if not isinstance(getattr(args, key), int) or getattr(args, key) < 0:
            raise ValueError(f"{key} must be a non-negative integer")
    args.data_dir = str(
        resolve_data_dir(
            configs["data"].get("output_dir"),
            configs["data"].get("output_name"),
        )
    )
    args.output_dir = args.output_dir or os.path.join(base_dir(), "checkpoints", args.run)
    branching = cli.branch_from is not None or cli.branch_step is not None
    if (cli.branch_from is None) != (cli.branch_step is None):
        raise ValueError("--branch-from and --branch-step must be provided together")
    if args.resume is not None and branching:
        raise ValueError("--resume and --branch-from are mutually exclusive")
    if not branching and cli.branch_schedule != "inherit":
        raise ValueError("--branch-schedule new requires --branch-from")
    if args.resume is None and latest(args.output_dir) is not None:
        raise FileExistsError(f"checkpoints already exist: {args.output_dir}; pass --resume STEP")
    metadata = load_metadata(args.output_dir, args.resume) if args.resume is not None else None
    parent_directory = cli.branch_from.expanduser().resolve() if branching else None
    if parent_directory == Path(args.output_dir).expanduser().resolve():
        raise ValueError("branch output directory must differ from its parent")
    parent_metadata = load_metadata(parent_directory, cli.branch_step) if branching else None
    if metadata:
        args.global_token_offset = metadata["resolved"].get("global_token_offset", 0)
    data_token_offset = metadata["resolved"].get("data_token_offset", 0) if metadata else 0
    rank, local_rank, world_size, device = init_runtime(args.device)
    distributed = world_size > 1
    master = rank == 0
    parent = metadata["resolved"].get("parent_checkpoint") if metadata else None
    branch_schedule = metadata["resolved"].get("branch_schedule") if metadata else None
    if parent_metadata:
        objects = [checkpoint_identity(parent_directory, cli.branch_step) if master else None]
        if distributed:
            dist.broadcast_object_list(objects, src=0)
        parent = objects[0]
        branch_schedule = cli.branch_schedule
    tokenizer = get_tokenizer(**configs["tokenizer"])
    manifest = load_manifest(args.data_dir)
    manifest_hash = manifest_fingerprint(manifest)
    source_ids = tuple(source["id"] for source in manifest["sources"])
    if manifest["tokenizer"]["fingerprint"] != tokenizer.fingerprint():
        raise ValueError("dataset and tokenizer do not match")

    error: list[str | None] = [None]
    if master:
        try:
            verify_shards(args.data_dir, manifest)
        except Exception as exception:
            error[0] = str(exception)
    if distributed:
        dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise ValueError(error[0])

    model = build_model(
        configs["model"],
        tokenizer.vocab_size,
        tokenizer.bos_id,
        tokenizer.eos_id,
        loss_backend=args.loss_backend,
    ).to(device)
    config = model.config
    model.init_weights()
    parameters = tuple(model.parameters())
    optimizer = model.optimizer(args.lr, args.weight_decay, args.optimizer)
    update_monitor = build_update_monitor(model, optimizer)

    batch_limit = args.device_batch_size if cli.device_batch_size is None else cli.device_batch_size
    args.device_batch_size = resolve_device_batch_size(
        batch_limit,
        args.batch_tokens,
        args.sequence_length,
        world_size,
    )
    micro_tokens = args.device_batch_size * args.sequence_length * world_size
    accumulation = args.batch_tokens // micro_tokens
    steps = math.ceil(args.train_tokens / args.batch_tokens)
    consumed_tokens = steps * args.batch_tokens
    schedule_step_offset = 0
    schedule_steps = steps
    if metadata:
        schedule_step_offset = metadata["resolved"].get("schedule_step_offset", 0)
        schedule_steps = metadata["resolved"].get("schedule_steps", steps)
    elif parent_metadata:
        (
            args.global_token_offset,
            data_token_offset,
            schedule_step_offset,
            schedule_steps,
        ) = branch_position(
            parent_metadata,
            args.batch_tokens,
            steps if cli.branch_schedule == "inherit" else None,
        )
        if cli.branch_schedule == "new":
            schedule_step_offset = 0
            schedule_steps = steps
    if (
        not isinstance(args.global_token_offset, int)
        or isinstance(args.global_token_offset, bool)
        or args.global_token_offset < 0
        or args.global_token_offset % args.batch_tokens
    ):
        raise ValueError("global token offset must align with optimizer batches")
    global_step_offset = args.global_token_offset // args.batch_tokens
    global_consumed_tokens = args.global_token_offset + consumed_tokens
    milestones = checkpoint_milestones(
        args.checkpoint_tokens, args.batch_tokens, args.global_token_offset, steps
    )
    if manifest["splits"]["train"]["tokens"] <= consumed_tokens:
        raise ValueError("packed dataset is too small for this run")

    data_state = None
    start_step = 0
    elapsed_training = 0.0
    elapsed_optimizer = 0.0
    elapsed_evaluation = 0.0
    elapsed_active = 0.0
    elapsed_checkpoint = 0.0
    if args.resume is not None:
        model_state, optimizer_state, loaded_metadata = load(args.output_dir, args.resume, device)
        if loaded_metadata != metadata:
            raise ValueError("checkpoint metadata changed while loading")
        stored_config = ArchitectureConfig.from_dict(metadata["config"]).settings()
        if stored_config != config.settings() or metadata["manifest"] != manifest_hash:
            raise ValueError("checkpoint does not match the model or dataset")
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        start_step = metadata["step"]
        data_state = metadata["data_state"]
        validate_loader_progress(data_state, data_token_offset + start_step * args.batch_tokens)
        elapsed_training = metadata["training_seconds"]
        timing = load_timing(args.output_dir, args.resume) or metadata.get("timing", {})
        elapsed_optimizer = timing.get("optimizer_seconds", elapsed_training)
        elapsed_evaluation = timing.get("evaluation_seconds", 0.0)
        elapsed_active = timing.get("active_seconds", elapsed_training)
        elapsed_checkpoint = timing.get("checkpoint_seconds", 0.0)
    elif parent_metadata:
        stored_config = ArchitectureConfig.from_dict(parent_metadata["config"]).settings()
        if stored_config != config.settings() or parent_metadata["manifest"] != manifest_hash:
            raise ValueError("branch parent does not match the model or dataset")
        branch_settings = {**vars(args), "world_size": world_size}
        changed = changed_branch_settings(
            parent_metadata["resolved"],
            branch_settings,
            allow_schedule_change=cli.branch_schedule == "new",
        )
        if changed:
            raise ValueError(f"branch settings changed: {', '.join(changed)}")
        model_state, optimizer_state, loaded_parent = load(
            parent_directory, cli.branch_step, device
        )
        if loaded_parent != parent_metadata:
            raise ValueError("parent checkpoint metadata changed while loading")
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        data_state = parent_metadata["data_state"]
        validate_loader_progress(data_state, data_token_offset)

    dataset_provenance = {
        "format": manifest["format"],
        "requested_train_tokens": manifest["requested_train_tokens"],
        "mixture": manifest["mixture"],
        "sources": [
            {
                "id": source["id"],
                "repo": source["repo"],
                "revision": source["revision"],
                "file_list_sha256": source["file_list_sha256"],
            }
            for source in manifest["sources"]
        ],
    }
    resolved = {
        **vars(args),
        "experiment": str(Path(cli.experiment).resolve()),
        "tokenizer": configs["tokenizer"],
        "model": config.export(),
        "parameters": model.parameter_count(),
        "manifest": manifest_hash,
        "dataset": dataset_provenance,
        "world_size": world_size,
        "accumulation_steps": accumulation,
        "steps": steps,
        "consumed_tokens": consumed_tokens,
        "global_step_offset": global_step_offset,
        "global_consumed_tokens": global_consumed_tokens,
        "data_token_offset": data_token_offset,
        "schedule_step_offset": schedule_step_offset,
        "schedule_steps": schedule_steps,
        "parent_checkpoint": parent,
        "branch_schedule": branch_schedule,
        "milestone_steps": {str(step): token for step, token in milestones.items()},
        "update_monitor": list(update_monitor.names) if update_monitor else [],
    }
    if metadata:
        changed = changed_resume_settings(metadata["resolved"], resolved)
        if changed:
            raise ValueError(f"resume settings changed: {', '.join(changed)}")
    print0(json.dumps(resolved, indent=2, sort_keys=True))

    if master and args.run != "dummy":
        run = wandb.init(
            project=args.wandb_project,
            name=args.run,
            group=args.wandb_group,
            job_type=args.training_phase,
            id=metadata.get("wandb_id") if metadata else None,
            resume="must" if metadata and metadata.get("wandb_id") else None,
            config=resolved,
        )
        wandb.define_metric("progress/step")
        wandb.define_metric("*", step_metric="progress/step")
    else:
        run = NullRun()

    train_data = packed_loader(
        tokenizer,
        args.device_batch_size,
        args.sequence_length,
        "train",
        device=device,
        resume_state_dict=data_state,
        data_dir=args.data_dir,
    )
    inputs, targets, data_state = next(train_data)
    compiled_model: Any = (
        model
        if args.no_compile
        else torch.compile(model, dynamic=False, mode="max-autotune-no-cudagraphs")
    )
    train_model = compiled_model
    if distributed:
        train_model = DistributedDataParallel(
            train_model, device_ids=[local_rank], broadcast_buffers=False
        )
    flops = model.flops_per_token(args.sequence_length)

    def validation(step):
        nonlocal elapsed_evaluation
        started = time.perf_counter()
        tokens_per_step = args.device_batch_size * args.sequence_length * world_size
        eval_tokens = args.final_eval_tokens if step == steps else args.eval_tokens
        val_steps = max(1, min(eval_tokens, manifest["splits"]["val"]["tokens"]) // tokens_per_step)
        loader = packed_loader(
            tokenizer,
            args.device_batch_size,
            args.sequence_length,
            "val",
            device=device,
            data_dir=args.data_dir,
        )
        loss, source_losses = validate(compiled_model, loader, val_steps, world_size, source_ids)
        evaluated_tokens = val_steps * tokens_per_step
        elapsed_evaluation += time.perf_counter() - started
        global_step = global_step_offset + step
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
        run.log(metrics)
        print0(f"step {global_step:,} ({global_tokens:,} tokens) | validation loss {loss:.5f}")
        return loss, source_losses, evaluated_tokens

    def checkpoint(
        step,
        validation_loss,
        validation_source_losses,
        validation_step,
        validation_tokens,
        milestone,
    ):
        nonlocal elapsed_checkpoint
        started = time.perf_counter()
        if master:
            global_tokens = args.global_token_offset + step * args.batch_tokens
            state = {
                "step": step,
                "global_step": global_step_offset + step,
                "global_tokens": global_tokens,
                "training_phase": args.training_phase,
                "config": config.settings(),
                "resolved": resolved,
                "manifest": manifest_hash,
                "data_state": data_state,
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
                "training_seconds": elapsed_training,
                "timing": {
                    "optimizer_seconds": elapsed_optimizer,
                    "evaluation_seconds": elapsed_evaluation,
                    "checkpoint_seconds": elapsed_checkpoint,
                    "active_seconds": elapsed_active + time.perf_counter() - session_started,
                },
                "wandb_id": run.id,
            }
            save(
                args.output_dir,
                step,
                model.state_dict(),
                optimizer.state_dict(),
                state,
                timing=lambda: {
                    "optimizer_seconds": elapsed_optimizer,
                    "evaluation_seconds": elapsed_evaluation,
                    "checkpoint_seconds": elapsed_checkpoint + time.perf_counter() - started,
                    "active_seconds": elapsed_active + time.perf_counter() - session_started,
                },
            )
        if distributed:
            dist.barrier()
        elapsed_checkpoint += time.perf_counter() - started

    if metadata:
        validation_loss = metadata["validation_loss"]
        validation_source_losses = metadata.get("validation_source_losses", {})
        validation_step = metadata.get("validation_step")
        validation_tokens = metadata.get("validation_tokens", 0)
    else:
        validation_loss, validation_source_losses, validation_tokens = validation(0)
        validation_step = 0
    synchronize = torch.cuda.synchronize if device.type == "cuda" else lambda: None
    for step in range(start_step, steps):
        completed = step + 1
        should_log = completed == 1 or completed % args.log_every == 0
        update_snapshot = update_monitor.snapshot() if master and should_log else None
        synchronize()
        started = time.perf_counter()
        scale = lr_scale(
            schedule_step_offset + step,
            schedule_steps,
            args.warmup_steps,
            args.min_lr,
            args.lr_schedule,
            args.decay_steps,
        )
        loss_sum, grad_norm, batch = optimization_step(
            train_model,
            parameters,
            optimizer,
            train_data,
            (inputs, targets, data_state),
            accumulation,
            args.grad_clip,
            args.lr * scale,
            distributed,
        )
        inputs, targets, data_state = batch
        synchronize()
        duration = time.perf_counter() - started
        elapsed_optimizer += duration
        if completed > 10:
            elapsed_training += duration
        if distributed and should_log:
            dist.all_reduce(loss_sum, op=dist.ReduceOp.AVG)
        if should_log:
            metrics = {
                "progress/step": global_step_offset + completed,
                "progress/phase_step": completed,
                "progress/tokens": args.global_token_offset + completed * args.batch_tokens,
                "train/loss": loss_sum.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/grad_norm": float(grad_norm),
                "performance/tokens_per_second": args.batch_tokens / duration,
                "performance/tflops": flops * args.batch_tokens / duration / 1e12,
                "data/next_source": data_state["selected_source"],
                "data/next_source_epoch": data_state["source_epochs"][
                    data_state["selected_source"]
                ],
                "data/next_phase": data_state["phase"],
                "data/next_shard": data_state["shard"]["index"],
            }
            if update_snapshot is not None:
                for name, values in update_monitor.metrics(update_snapshot).items():
                    path = name.replace(".", "/")
                    for metric, value in values.items():
                        metrics[f"optimization/{path}/{metric}"] = value
            run.log(metrics)
            print0(
                f"step {metrics['progress/step']:,}/{global_step_offset + steps:,} | loss {metrics['train/loss']:.5f} | {metrics['performance/tokens_per_second']:,.0f} tok/s"
            )
        milestone = milestones.get(completed)
        if (
            (args.eval_every > 0 and completed % args.eval_every == 0)
            or milestone is not None
            or completed == steps
        ):
            validation_loss, validation_source_losses, validation_tokens = validation(completed)
            validation_step = completed
        if (
            (args.save_every > 0 and completed % args.save_every == 0)
            or milestone is not None
            or completed == steps
        ):
            checkpoint(
                completed,
                validation_loss,
                validation_source_losses,
                validation_step,
                validation_tokens,
                milestone,
            )

    run.finish()
    if master:
        summary = {
            "training_phase": args.training_phase,
            "steps": steps,
            "global_step": global_step_offset + steps,
            "global_tokens": global_consumed_tokens,
            "optimizer_seconds": elapsed_optimizer,
            "evaluation_seconds": elapsed_evaluation,
            "checkpoint_seconds": elapsed_checkpoint,
            "active_seconds": elapsed_active + time.perf_counter() - session_started,
        }
        path = Path(args.output_dir) / "run_summary.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    cleanup()


def main():
    cli = arguments()
    configs = load_experiment(cli.experiment, "data", "tokenizer", "model", "train")
    train(configs, cli)


if __name__ == "__main__":
    main()
