"""Fine-tune Speck on assistant-masked packed chat data."""

import argparse
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

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
        default="experiments/Speck1-140M-Instruct",
        help="experiment directory (default: %(default)s)",
    )
    parser.add_argument(
        "--device", default=None, help="training device; defaults to automatic runtime selection"
    )
    parser.add_argument(
        "--resume", type=int, default=None, help="SFT checkpoint step to resume from"
    )
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
            "SFT batch-token count, device batch size, epoch count, logging interval, sequence "
            "length, and checkpoint retention must be positive"
        )
    if args.eval_every < 0 or args.save_every < 0 or args.warmup_steps < 0:
        raise ValueError("SFT step intervals must not be negative")
    if (
        not isinstance(args.sequence_lengths, list)
        or sorted(set(args.sequence_lengths)) != args.sequence_lengths
        or args.sequence_lengths[-1] != args.sequence_length
    ):
        raise ValueError(
            "SFT sequence lengths must be unique, in ascending order, and end at sequence_length"
        )
    if args.lr <= 0 or args.weight_decay < 0 or args.grad_clip <= 0:
        raise ValueError("invalid SFT optimization settings")
    if not 0 <= args.min_lr <= 1:
        raise ValueError("SFT min_lr must be a multiplier between zero and one")
    return args


class SFTTrainer:
    """Own one supervised fine-tuning lifecycle and its mutable execution state."""

    def __init__(self, configs, cli):
        self.configs = configs
        self.cli = cli
        self.tracking: Any = None
        args = _settings(configs["sft"])
        args.device = cli.device
        args.resume = cli.resume
        args.no_compile = cli.no_compile
        args.data_dir = str(resolve_sft_data_dir(args.dataset, args.data_dir))
        args.output_dir = args.output_dir or os.path.join(base_dir(), "checkpoints", args.run)
        if args.resume is None and latest(args.output_dir) is not None:
            raise FileExistsError(
                f"checkpoints already exist: {args.output_dir}; pass --resume STEP"
            )
        self.args = args

    def _initialize_runtime(self):
        self.rank, self.local_rank, self.world_size, self.device = init_runtime(self.args.device)
        self.distributed = self.world_size > 1
        self.master = self.rank == 0

    def _load_and_verify_data(self):
        args = self.args
        self.tokenizer = get_chat_tokenizer(**self.configs["tokenizer"])
        self.manifest = load_sft_manifest(args.data_dir)
        self.manifest_hash = manifest_fingerprint(self.manifest)
        if self.manifest["tokenizer"] != self.tokenizer.metadata():
            raise ValueError("SFT dataset and tokenizer do not match")
        expected_dataset = {**args.dataset, "sequence_lengths": args.sequence_lengths}
        if self.manifest["dataset"] != expected_dataset:
            raise ValueError("prepared SFT dataset does not match the configured dataset")
        error: list[str | None] = [None]
        if self.master:
            try:
                verify_sft_dataset(args.data_dir, self.manifest)
            except Exception as exception:
                error[0] = str(exception)
        if self.distributed:
            dist.broadcast_object_list(error, src=0)
        if error[0]:
            raise ValueError(error[0])

    def _resolve_geometry(self):
        args = self.args
        micro_tokens = args.device_batch_size * args.sequence_length * self.world_size
        if args.batch_tokens % micro_tokens:
            raise ValueError("SFT batch tokens must be divisible by the distributed microbatch")
        self.accumulation = args.batch_tokens // micro_tokens
        self.device_tokens = args.device_batch_size * args.sequence_length
        self.train_plan = sft_plan(
            self.manifest,
            "train",
            self.device_tokens,
            self.world_size,
            self.accumulation,
        )
        self.steps_per_epoch = self.train_plan["cycle_microbatches"] // self.accumulation
        self.steps = self.steps_per_epoch * args.epochs

    def _initialize_model(self):
        args = self.args
        self.data_state = None
        self.start_step = 0
        self.trained_supervised_tokens = 0
        self.elapsed_training = 0.0
        self.metadata = None
        checkpoint_state = None
        if args.resume is not None:
            checkpoint_state = load(args.output_dir, args.resume, self.device)
            self.metadata = checkpoint_state[2]
            self.model, self.config, self.pretrained = self._resumed_model()
        else:
            self.model = build_model(
                self.configs["model"],
                self.tokenizer.base.vocab_size,
                self.tokenizer.bos_id,
                self.tokenizer.eos_id,
            )
            self.pretrained = load_pretrained(self.model, **args.pretrained)
            self.model.resize_token_embeddings(self.tokenizer.vocab_size)
            self.config = self.model.config
        self.model = self.model.to(self.device)
        self.parameters = tuple(self.model.parameters())
        self.optimizer = self.model.optimizer(args.lr, args.weight_decay, args.optimizer)
        if checkpoint_state is not None:
            self._restore_checkpoint_state(checkpoint_state)

    def _resumed_model(self):
        metadata = self.metadata
        if metadata is None:
            raise RuntimeError("resume metadata was not loaded")
        if metadata.get("training_phase") != "sft" or metadata["manifest"] != self.manifest_hash:
            raise ValueError("SFT checkpoint does not match the model or dataset")
        pretrained = metadata["resolved"]["pretrained"]
        source = {key: pretrained[key] for key in ("repo", "revision", "filename")}
        if source != self.args.pretrained:
            raise ValueError("SFT checkpoint uses a different pretrained model")
        config = ArchitectureConfig.from_dict(metadata["config"])
        expected_model = dict(self.configs["model"])
        expected_model.pop("expected_parameters", None)
        expected_model.update(
            vocab_size=self.tokenizer.vocab_size,
            bos_token_id=self.tokenizer.bos_id,
            eos_token_id=self.tokenizer.eos_id,
        )
        if config.settings() != ArchitectureConfig.from_dict(expected_model).settings():
            raise ValueError("SFT checkpoint architecture does not match the experiment")
        return SpeckForCausalLM(config), config, pretrained

    def _restore_checkpoint_state(self, checkpoint_state):
        metadata = self.metadata
        if metadata is None:
            raise RuntimeError("resume metadata was not loaded")
        model_state, optimizer_state, _ = checkpoint_state
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)
        self.start_step = metadata["step"]
        self.data_state = metadata["data_state"]
        if (
            self.data_state.get("global_consumed_microbatches")
            != self.start_step * self.accumulation
        ):
            raise ValueError("SFT checkpoint loader position does not match its step")
        self.trained_supervised_tokens = metadata["trained_supervised_tokens"]
        self.elapsed_training = metadata["training_seconds"]

    def _build_resolved_settings(self):
        self.resolved = {
            **vars(self.args),
            "experiment": str(Path(self.cli.experiment).resolve()),
            "tokenizer": self.tokenizer.metadata(),
            "model": self.config.export(),
            "parameters": self.model.parameter_count(),
            "pretrained": self.pretrained,
            "manifest": self.manifest_hash,
            "dataset": self.manifest["dataset"],
            "world_size": self.world_size,
            "accumulation_steps": self.accumulation,
            "steps_per_epoch": self.steps_per_epoch,
            "steps": self.steps,
            "device_tokens": self.device_tokens,
            "bucket_plan": {
                "real_microbatches": self.train_plan["real_microbatches"],
                "dummy_microbatches": self.train_plan["dummy_microbatches"],
                "cycle_microbatches": self.train_plan["cycle_microbatches"],
                "context_tokens": self.train_plan["context_tokens"],
                "buckets": self.train_plan["buckets"],
                "fingerprint": self.train_plan["fingerprint"],
            },
        }
        if self.metadata:
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
            changed = [
                key
                for key in immutable
                if self.metadata["resolved"].get(key) != self.resolved.get(key)
            ]
            if changed:
                raise ValueError(f"SFT resume settings changed: {', '.join(changed)}")
        print0(json.dumps(self.resolved, indent=2, sort_keys=True))

    def _publish_tokenizer(self):
        if self.master:
            self.tokenizer.save_pretrained(
                Path(self.args.output_dir) / "tokenizer",
                self.config.max_position_embeddings,
            )
        if self.distributed:
            dist.barrier()

    def _initialize_tracking(self):
        args = self.args
        if self.master and args.run != "dummy":
            self.tracking = wandb.init(
                project=args.wandb_project,
                name=args.run,
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
        self.train_data = sft_loader(
            self.tokenizer,
            self.device_tokens,
            self.accumulation,
            device=cast(Any, self.device),
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
                mode="max-autotune-no-cudagraphs",
            )
        )

    def _validate(self, step):
        validation_plan = sft_plan(
            self.manifest,
            "val",
            self.device_tokens,
            self.world_size,
        )
        loader = sft_loader(
            self.tokenizer,
            self.device_tokens,
            split="val",
            device=cast(Any, self.device),
            data_dir=self.args.data_dir,
        )
        loss, supervised = validate_sft(
            self.train_model,
            loader,
            validation_plan["cycle_microbatches"],
            self.distributed,
        )
        self.tracking.log(
            {
                "progress/step": step,
                "progress/tokens": step * self.args.batch_tokens,
                "validation/loss": loss,
                "validation/perplexity": math.exp(min(loss, 20)),
                "validation/supervised_tokens": supervised,
            }
        )
        print0(f"step {step:,} | validation loss {loss:.5f}")
        return loss

    def _checkpoint(self, step, validation_loss):
        if self.master:
            state = {
                "format_version": 1,
                "training_phase": "sft",
                "step": step,
                "config": self.config.settings(),
                "resolved": self.resolved,
                "manifest": self.manifest_hash,
                "data_state": self.data_state,
                "trained_supervised_tokens": self.trained_supervised_tokens,
                "validation_loss": validation_loss,
                "training_seconds": self.elapsed_training,
                "wandb_id": self.tracking.id,
            }
            save(
                self.args.output_dir,
                step,
                self.model.state_dict(),
                self.optimizer.state_dict(),
                state,
            )
            prune(self.args.output_dir, self.args.keep_checkpoints)
        if self.distributed:
            dist.barrier()

    def _run_steps(self):
        args = self.args
        validation_loss = (
            self.metadata.get("validation_loss") if self.metadata else self._validate(0)
        )
        synchronize = torch.cuda.synchronize if self.device.type == "cuda" else lambda: None
        pending_supervised_tokens = torch.zeros((), device=self.device, dtype=torch.long)
        timing_started = time.perf_counter()
        timing_steps = 0
        for step in range(self.start_step, self.steps):
            completed = step + 1
            should_log = completed == 1 or completed % args.log_every == 0
            should_validate = (
                args.eval_every > 0 and completed % args.eval_every == 0
            ) or completed == self.steps
            should_save = (
                args.save_every > 0 and completed % args.save_every == 0
            ) or completed == self.steps
            scale = lr_scale(step, self.steps, args.warmup_steps, args.min_lr)
            loss, grad_norm, batch, supervised = sft_optimization_step(
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
            pending_supervised_tokens += supervised
            timing_steps += 1
            should_flush_timing = should_log or should_validate or should_save or completed == 10
            duration = None
            if should_flush_timing:
                synchronize()
                window_duration = time.perf_counter() - timing_started
                if completed > 10:
                    self.elapsed_training += window_duration
                duration = window_duration / timing_steps
                self.trained_supervised_tokens += int(pending_supervised_tokens.item())
                pending_supervised_tokens.zero_()
            if should_log:
                self._log_step(completed, loss, grad_norm, duration)
            if should_validate:
                validation_loss = self._validate(completed)
            if should_save:
                self._checkpoint(completed, validation_loss)
            if should_flush_timing:
                timing_started = time.perf_counter()
                timing_steps = 0

    def _log_step(self, completed, loss, grad_norm, duration):
        assert duration is not None
        data_state = self.data_state
        if data_state is None:
            raise RuntimeError("training data state is unavailable")
        metrics = {
            "progress/step": completed,
            "progress/epoch": completed / self.steps_per_epoch,
            "progress/tokens": completed * self.args.batch_tokens,
            "progress/supervised_tokens": self.trained_supervised_tokens,
            "train/loss": loss.item(),
            "train/lr": float(self.optimizer.param_groups[0]["lr"]),
            "train/grad_norm": float(grad_norm),
            "performance/tokens_per_second": self.args.batch_tokens / duration,
            "data/next_sequence_length": data_state["sequence_length"],
        }
        self.tracking.log(metrics)
        print0(
            f"step {completed:,}/{self.steps:,} | loss {metrics['train/loss']:.5f} | "
            f"{metrics['performance/tokens_per_second']:,.0f} tok/s"
        )

    def _finish_tracking(self):
        if self.tracking is None:
            return
        tracking, self.tracking = self.tracking, None
        tracking.finish()

    def run(self):
        try:
            self._initialize_runtime()
            self._load_and_verify_data()
            self._resolve_geometry()
            self._initialize_model()
            self._build_resolved_settings()
            self._publish_tokenizer()
            self._initialize_tracking()
            self._prepare_execution()
            self._run_steps()
            self._finish_tracking()
        finally:
            self._finish_tracking()
            cleanup()


def main():
    cli = arguments()
    configs = load_experiment(cli.experiment, "tokenizer", "model", "sft")
    SFTTrainer(configs, cli).run()


if __name__ == "__main__":
    main()
