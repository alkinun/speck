"""Instruction tune Speck on assistant-masked packed chat data."""

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
from speck.chat import get_chat_tokenizer
from speck.checkpoint import latest, load, prune, save
from speck.common import NullRun, base_dir, cleanup, init_runtime, print0
from speck.config import load_experiment
from speck.dataloader import manifest_fingerprint
from speck.model import SpeckForCausalLM, build_model
from speck.pretrained import load_pretrained
from speck.sft import (
    load_sft_manifest,
    resolve_sft_data_dir,
    sft_loader,
    sft_optimization_step,
    sft_plan,
    validate_sft,
    verify_sft_dataset,
)
from speck.train import lr_scale


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment",
        nargs="?",
        default="experiments/Speck1-140M",
        help="experiment directory (default: %(default)s)",
    )
    parser.add_argument("--device", default=None, help="training device")
    parser.add_argument("--resume", type=int, default=None, help="SFT checkpoint step to resume")
    parser.add_argument("--no-compile", action="store_true", help="disable torch.compile")
    return parser.parse_args()


def _settings(value):
    required = {
        "batch_tokens",
        "data_dir",
        "dataset",
        "device_batch_size",
        "epochs",
        "eval_every",
        "grad_clip",
        "log_every",
        "lr",
        "keep_checkpoints",
        "min_lr",
        "optimizer",
        "output_dir",
        "pretrained",
        "run",
        "save_every",
        "sequence_length",
        "sequence_lengths",
        "wandb_project",
        "warmup_steps",
        "weight_decay",
    }
    if set(value) != required:
        missing = sorted(required - set(value))
        unknown = sorted(set(value) - required)
        raise ValueError(f"invalid SFT settings; missing={missing}, unknown={unknown}")
    args = SimpleNamespace(**value)
    integer_positive = (
        "batch_tokens",
        "device_batch_size",
        "epochs",
        "log_every",
        "sequence_length",
        "keep_checkpoints",
    )
    if any(
        not isinstance(getattr(args, key), int) or getattr(args, key) < 1
        for key in integer_positive
    ):
        raise ValueError(
            "SFT batch, epoch, evaluation, logging, and sequence values must be positive"
        )
    if args.eval_every < 0 or args.save_every < 0 or args.warmup_steps < 0:
        raise ValueError("SFT step intervals must not be negative")
    if (
        not isinstance(args.sequence_lengths, list)
        or sorted(set(args.sequence_lengths)) != args.sequence_lengths
        or args.sequence_lengths[-1] != args.sequence_length
    ):
        raise ValueError("SFT sequence lengths must be unique, ordered, and end at sequence_length")
    if args.lr <= 0 or args.weight_decay < 0 or args.grad_clip <= 0:
        raise ValueError("invalid SFT optimization settings")
    if not 0 <= args.min_lr <= 1:
        raise ValueError("SFT min_lr must be a multiplier between zero and one")
    return args


def main():
    cli = arguments()
    configs = load_experiment(cli.experiment, "tokenizer", "model", "sft")
    args = _settings(configs["sft"])
    args.device = cli.device
    args.resume = cli.resume
    args.no_compile = cli.no_compile
    args.data_dir = str(resolve_sft_data_dir(args.dataset, args.data_dir))
    args.output_dir = args.output_dir or os.path.join(base_dir(), "checkpoints", args.run)
    if args.resume is None and latest(args.output_dir) is not None:
        raise FileExistsError(f"checkpoints already exist: {args.output_dir}; pass --resume")

    rank, local_rank, world_size, device = init_runtime(args.device)
    distributed = world_size > 1
    master = rank == 0
    tokenizer = get_chat_tokenizer(**configs["tokenizer"])
    manifest = load_sft_manifest(args.data_dir)
    manifest_hash = manifest_fingerprint(manifest)
    if manifest["tokenizer"] != tokenizer.metadata():
        raise ValueError("SFT dataset and tokenizer do not match")
    expected_dataset = {**args.dataset, "sequence_lengths": args.sequence_lengths}
    if manifest["dataset"] != expected_dataset:
        raise ValueError("prepared SFT dataset does not match the configured dataset")
    error: list[str | None] = [None]
    if master:
        try:
            verify_sft_dataset(args.data_dir, manifest)
        except Exception as exception:
            error[0] = str(exception)
    if distributed:
        dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise ValueError(error[0])

    micro_tokens = args.device_batch_size * args.sequence_length * world_size
    if args.batch_tokens % micro_tokens:
        raise ValueError("SFT batch tokens must be divisible by the distributed microbatch")
    accumulation = args.batch_tokens // micro_tokens
    device_tokens = args.device_batch_size * args.sequence_length
    train_plan = sft_plan(manifest, "train", device_tokens, world_size, accumulation)
    steps_per_epoch = train_plan["cycle_microbatches"] // accumulation
    steps = steps_per_epoch * args.epochs

    data_state = None
    start_step = 0
    trained_supervised_tokens = 0
    elapsed_training = 0.0
    metadata = None
    checkpoint_state = None
    if args.resume is not None:
        checkpoint_state = load(args.output_dir, args.resume, device)
        metadata = checkpoint_state[2]
        if metadata.get("training_phase") != "sft" or metadata["manifest"] != manifest_hash:
            raise ValueError("SFT checkpoint does not match the model or dataset")
        pretrained = metadata["resolved"]["pretrained"]
        source = {key: pretrained[key] for key in ("repo", "revision", "filename")}
        if source != args.pretrained:
            raise ValueError("SFT checkpoint uses a different pretrained model")
        config = ArchitectureConfig.from_dict(metadata["config"])
        expected_model = dict(configs["model"])
        expected_model.pop("expected_parameters", None)
        expected_model.update(
            vocab_size=tokenizer.vocab_size,
            bos_token_id=tokenizer.bos_id,
            eos_token_id=tokenizer.eos_id,
        )
        if config.settings() != ArchitectureConfig.from_dict(expected_model).settings():
            raise ValueError("SFT checkpoint architecture does not match the experiment")
        model = SpeckForCausalLM(config)
    else:
        model = build_model(
            configs["model"], tokenizer.base.vocab_size, tokenizer.bos_id, tokenizer.eos_id
        )
        pretrained = load_pretrained(model, **args.pretrained)
        model.resize_token_embeddings(tokenizer.vocab_size)
        config = model.config
    model = model.to(device)
    parameters = tuple(model.parameters())
    optimizer = model.optimizer(args.lr, args.weight_decay, args.optimizer)
    if checkpoint_state is not None:
        model_state, optimizer_state, _ = checkpoint_state
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        start_step = metadata["step"]
        data_state = metadata["data_state"]
        if data_state.get("global_consumed_microbatches") != start_step * accumulation:
            raise ValueError("SFT checkpoint loader position does not match its step")
        trained_supervised_tokens = metadata["trained_supervised_tokens"]
        elapsed_training = metadata["training_seconds"]

    resolved = {
        **vars(args),
        "experiment": str(Path(cli.experiment).resolve()),
        "tokenizer": tokenizer.metadata(),
        "model": config.export(),
        "parameters": model.parameter_count(),
        "pretrained": pretrained,
        "manifest": manifest_hash,
        "dataset": manifest["dataset"],
        "world_size": world_size,
        "accumulation_steps": accumulation,
        "steps_per_epoch": steps_per_epoch,
        "steps": steps,
        "device_tokens": device_tokens,
        "bucket_plan": {
            "real_microbatches": train_plan["real_microbatches"],
            "dummy_microbatches": train_plan["dummy_microbatches"],
            "cycle_microbatches": train_plan["cycle_microbatches"],
            "context_tokens": train_plan["context_tokens"],
            "buckets": train_plan["buckets"],
            "fingerprint": train_plan["fingerprint"],
        },
    }
    if metadata:
        immutable = (
            "sequence_length",
            "device_batch_size",
            "batch_tokens",
            "epochs",
            "lr",
            "weight_decay",
            "warmup_steps",
            "min_lr",
            "grad_clip",
            "optimizer",
            "world_size",
            "manifest",
            "pretrained",
        )
        changed = [key for key in immutable if metadata["resolved"].get(key) != resolved.get(key)]
        if changed:
            raise ValueError(f"SFT resume settings changed: {', '.join(changed)}")
    print0(json.dumps(resolved, indent=2, sort_keys=True))

    if master:
        tokenizer.save_pretrained(
            Path(args.output_dir) / "tokenizer", config.max_position_embeddings
        )
    if distributed:
        dist.barrier()
    if master and args.run != "dummy":
        run = wandb.init(
            project=args.wandb_project,
            name=args.run,
            id=metadata.get("wandb_id") if metadata else None,
            resume="must" if metadata and metadata.get("wandb_id") else None,
            config=resolved,
        )
        wandb.define_metric("progress/step")
        wandb.define_metric("*", step_metric="progress/step")
    else:
        run = NullRun()

    train_data = sft_loader(
        tokenizer,
        device_tokens,
        accumulation,
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

    def validation(step):
        validation_plan = sft_plan(manifest, "val", device_tokens, world_size)
        loader = sft_loader(
            tokenizer,
            device_tokens,
            split="val",
            device=device,
            data_dir=args.data_dir,
        )
        loss, supervised = validate_sft(
            compiled_model,
            loader,
            validation_plan["cycle_microbatches"],
            distributed,
        )
        run.log(
            {
                "progress/step": step,
                "progress/tokens": step * args.batch_tokens,
                "validation/loss": loss,
                "validation/perplexity": math.exp(min(loss, 20)),
                "validation/supervised_tokens": supervised,
            }
        )
        print0(f"step {step:,} | validation loss {loss:.5f}")
        return loss

    def checkpoint(step, validation_loss):
        if master:
            state = {
                "format_version": 1,
                "training_phase": "sft",
                "step": step,
                "config": config.settings(),
                "resolved": resolved,
                "manifest": manifest_hash,
                "data_state": data_state,
                "trained_supervised_tokens": trained_supervised_tokens,
                "validation_loss": validation_loss,
                "training_seconds": elapsed_training,
                "wandb_id": run.id,
            }
            save(args.output_dir, step, model.state_dict(), optimizer.state_dict(), state)
            prune(args.output_dir, args.keep_checkpoints)
        if distributed:
            dist.barrier()

    validation_loss = metadata.get("validation_loss") if metadata else validation(0)
    synchronize = torch.cuda.synchronize if device.type == "cuda" else lambda: None
    for step in range(start_step, steps):
        synchronize()
        started = time.perf_counter()
        scale = lr_scale(step, steps, args.warmup_steps, args.min_lr)
        loss, grad_norm, batch, supervised = sft_optimization_step(
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
        trained_supervised_tokens += supervised
        synchronize()
        duration = time.perf_counter() - started
        completed = step + 1
        if completed > 10:
            elapsed_training += duration
        if completed == 1 or completed % args.log_every == 0:
            metrics = {
                "progress/step": completed,
                "progress/epoch": completed / steps_per_epoch,
                "progress/tokens": completed * args.batch_tokens,
                "progress/supervised_tokens": trained_supervised_tokens,
                "train/loss": loss.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/grad_norm": float(grad_norm),
                "performance/tokens_per_second": args.batch_tokens / duration,
                "data/next_sequence_length": data_state["sequence_length"],
            }
            run.log(metrics)
            print0(
                f"step {completed:,}/{steps:,} | loss {metrics['train/loss']:.5f} | "
                f"{metrics['performance/tokens_per_second']:,.0f} tok/s"
            )
        if (args.eval_every > 0 and completed % args.eval_every == 0) or completed == steps:
            validation_loss = validation(completed)
        if (args.save_every > 0 and completed % args.save_every == 0) or completed == steps:
            checkpoint(completed, validation_loss)

    run.finish()
    cleanup()


if __name__ == "__main__":
    main()
