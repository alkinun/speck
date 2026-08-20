"""distributed pretraining with validation, checkpoints, and wandb."""

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel

from speck.checkpoint import latest, load, save
from speck.common import NullRun, base_dir, cleanup, init_runtime, print0
from speck.config import load_experiment
from speck.dataloader import manifest_fingerprint, packed_loader
from speck.dataset import default_data_dir, load_manifest, verify_shards
from speck.hub import upload
from speck.model import build_model
from speck.tokenizer import get_tokenizer
from speck.train import lr_scale, optimization_step


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", nargs="?", default="experiments/speck00-200m")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", type=int, default=None)
    parser.add_argument("--no-compile", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def validate(model, loader, steps, world_size):
    model.eval()
    loss = torch.zeros((), device=next(model.parameters()).device)
    for _ in range(steps):
        inputs, targets, _ = next(loader)
        loss += model(inputs, targets)
    loss /= steps
    if world_size > 1:
        dist.all_reduce(loss, op=dist.ReduceOp.AVG)
    model.train()
    return loss.item()


def main():
    cli = arguments()
    configs = load_experiment(cli.experiment, "data", "tokenizer", "model", "train")
    args = SimpleNamespace(**configs["train"])
    args.device = cli.device
    args.resume = cli.resume
    args.no_compile = cli.no_compile
    args.data_dir = configs["data"].get("output_dir") or str(default_data_dir / "packed")
    args.output_dir = args.output_dir or os.path.join(base_dir(), "checkpoints", args.run)
    if args.resume is None and latest(args.output_dir) is not None:
        raise FileExistsError(f"checkpoints already exist: {args.output_dir}; pass --resume")
    rank, local_rank, world_size, device = init_runtime(args.device)
    distributed = world_size > 1
    master = rank == 0
    tokenizer = get_tokenizer(**configs["tokenizer"])
    manifest = load_manifest(args.data_dir)
    manifest_hash = manifest_fingerprint(manifest)
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
        configs["model"], tokenizer.vocab_size, tokenizer.bos_id, tokenizer.eos_id
    ).to(device)
    config = model.config
    model.init_weights()
    parameters = tuple(model.parameters())
    optimizer = model.optimizer(args.lr, args.weight_decay, args.optimizer)

    micro_tokens = args.device_batch_size * args.sequence_length * world_size
    if args.batch_tokens % micro_tokens:
        raise ValueError("batch tokens must be divisible by the distributed micro batch")
    accumulation = args.batch_tokens // micro_tokens
    steps = math.ceil(args.train_tokens / args.batch_tokens)
    consumed_tokens = steps * args.batch_tokens
    if manifest["splits"]["train"]["tokens"] <= consumed_tokens:
        raise ValueError("packed dataset is too small for this run")

    data_state = None
    start_step = 0
    elapsed_training = 0.0
    metadata = None
    if args.resume is not None:
        model_state, optimizer_state, metadata = load(args.output_dir, args.resume, device)
        if metadata["config"] != asdict(config) or metadata["manifest"] != manifest_hash:
            raise ValueError("checkpoint does not match the model or dataset")
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        start_step = metadata["step"]
        data_state = metadata["data_state"]
        elapsed_training = metadata["training_seconds"]

    resolved = {
        **vars(args),
        "experiment": str(Path(cli.experiment).resolve()),
        "tokenizer": configs["tokenizer"],
        "model": config.export(),
        "parameters": model.parameter_count(),
        "manifest": manifest_hash,
        "dataset": manifest["dataset"],
        "world_size": world_size,
        "accumulation_steps": accumulation,
        "steps": steps,
        "consumed_tokens": consumed_tokens,
    }
    if metadata:
        immutable = ("sequence_length", "device_batch_size", "batch_tokens", "train_tokens", "lr", "weight_decay", "warmup_steps", "min_lr", "grad_clip", "optimizer", "world_size")
        changed = [key for key in immutable if metadata["resolved"].get(key) != resolved.get(key)]
        if changed:
            raise ValueError(f"resume settings changed: {', '.join(changed)}")
    print0(json.dumps(resolved, indent=2, sort_keys=True))

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
    compiled_model: Any = model if args.no_compile else torch.compile(
        model, dynamic=False, mode="max-autotune-no-cudagraphs"
    )
    train_model = compiled_model
    if distributed:
        train_model = DistributedDataParallel(
            train_model, device_ids=[local_rank], broadcast_buffers=False
        )
    flops = model.flops_per_token(args.sequence_length)

    def validation(step):
        tokens_per_step = args.device_batch_size * args.sequence_length * world_size
        eval_tokens = args.final_eval_tokens if step == steps else args.eval_tokens
        val_steps = max(1, min(eval_tokens, manifest["splits"]["val"]["tokens"]) // tokens_per_step)
        loader = packed_loader(tokenizer, args.device_batch_size, args.sequence_length, "val", device=device, data_dir=args.data_dir)
        loss = validate(compiled_model, loader, val_steps, world_size)
        run.log({"progress/step": step, "progress/tokens": step * args.batch_tokens, "validation/loss": loss, "validation/perplexity": math.exp(min(loss, 20))})
        print0(f"step {step:,} | validation loss {loss:.5f}")
        return loss

    def checkpoint(step, validation_loss):
        if master:
            state = {
                "step": step,
                "config": asdict(config),
                "resolved": resolved,
                "manifest": manifest_hash,
                "data_state": data_state,
                "validation_loss": validation_loss,
                "training_seconds": elapsed_training,
                "wandb_id": run.id,
            }
            save(args.output_dir, step, model.state_dict(), optimizer.state_dict(), state)
            if args.hf_repo:
                commit_url = upload(
                    args.hf_repo,
                    args.output_dir,
                    step,
                    state,
                    private=args.hf_private,
                    optimizer=args.hf_upload_optimizer,
                )
                run.log({"progress/step": step, "artifacts/hf_commit": commit_url})
                print0(f"uploaded checkpoint: {commit_url}")
        if distributed:
            dist.barrier()

    validation_loss = metadata.get("validation_loss") if metadata else validation(0)
    synchronize = torch.cuda.synchronize if device.type == "cuda" else lambda: None
    for step in range(start_step, steps):
        synchronize()
        started = time.perf_counter()
        scale = lr_scale(step, steps, args.warmup_steps, args.min_lr)
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
        completed = step + 1
        if completed > 10:
            elapsed_training += duration
        should_log = completed == 1 or completed % args.log_every == 0
        if distributed and should_log:
            dist.all_reduce(loss_sum, op=dist.ReduceOp.AVG)
        if should_log:
            metrics = {
                "progress/step": completed,
                "progress/tokens": completed * args.batch_tokens,
                "train/loss": loss_sum.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/grad_norm": float(grad_norm),
                "performance/tokens_per_second": args.batch_tokens / duration,
                "performance/tflops": flops * args.batch_tokens / duration / 1e12,
                "data/epoch": data_state["epoch"],
                "data/shard": data_state["shard"],
            }
            run.log(metrics)
            print0(f"step {completed:,}/{steps:,} | loss {metrics['train/loss']:.5f} | {metrics['performance/tokens_per_second']:,.0f} tok/s")
        if (args.eval_every > 0 and completed % args.eval_every == 0) or completed == steps:
            validation_loss = validation(completed)
        if (args.save_every > 0 and completed % args.save_every == 0) or completed == steps:
            checkpoint(completed, validation_loss)

    run.finish()
    cleanup()


if __name__ == "__main__":
    main()
