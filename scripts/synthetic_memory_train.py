"""Train a controlled Speck mixer on deterministic synthetic memory tasks."""

import argparse
import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from speck.architecture import (
    ArchitectureConfig,
    BlockConfig,
    BlockGroup,
    GatedDeltaNetSpec,
    KimiDeltaAttentionSpec,
    StageConfig,
    SwiGLUSpec,
)
from speck.model import SpeckForCausalLM
from speck.synthetic import IGNORE_INDEX, mqar_batch, palindrome_batch, stack_batch

VARIANTS = ("gdn-silu", "gdn-sigmoid", "kda-sigmoid")
TASKS = ("mqar", "palindrome", "stack")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, default="mqar")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--sequence-length", type=int, default=1_024)
    parser.add_argument("--vocab-size", type=int, default=8_192)
    parser.add_argument("--num-pairs", type=int, default=256)
    parser.add_argument("--num-stacks", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-examples", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-seed", type=int, default=1_000_042)
    parser.add_argument("--early-stop-accuracy", type=float, default=0.99)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def positive_integer(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def resolved_settings(args):
    for name in (
        "sequence_length",
        "vocab_size",
        "batch_size",
        "eval_batch_size",
        "eval_examples",
        "max_steps",
        "eval_every",
        "log_every",
    ):
        positive_integer(getattr(args, name), name.replace("_", " "))
    if args.eval_examples % args.eval_batch_size:
        raise ValueError("evaluation examples must be divisible by evaluation batch size")
    for name in ("lr", "grad_clip"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name.replace('_', ' ')} must be positive and finite")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight decay must be non-negative and finite")
    if (
        not math.isfinite(args.early_stop_accuracy)
        or not 0 < args.early_stop_accuracy <= 1
    ):
        raise ValueError("early-stop accuracy must be in (0, 1]")
    if args.task == "mqar":
        positive_integer(args.num_pairs, "number of key-value pairs")
    if args.task == "stack":
        positive_integer(args.num_stacks, "number of stacks")
    return {
        "task": args.task,
        "variant": args.variant,
        "sequence_length": args.sequence_length,
        "vocab_size": args.vocab_size,
        "num_pairs": args.num_pairs,
        "num_stacks": args.num_stacks,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "eval_examples": args.eval_examples,
        "max_steps": args.max_steps,
        "eval_every": args.eval_every,
        "log_every": args.log_every,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "seed": args.seed,
        "validation_seed": args.validation_seed,
        "early_stop_accuracy": args.early_stop_accuracy,
        "device": args.device,
        "compiled": not args.no_compile,
    }


def architecture_for_variant(variant, sequence_length, vocab_size):
    if variant not in VARIANTS:
        raise ValueError(f"unknown synthetic-memory variant: {variant}")
    if variant == "kda-sigmoid":
        mixer = KimiDeltaAttentionSpec(128, 128, 2, 2, conv_kernel_size=4)
    else:
        mixer = GatedDeltaNetSpec(
            128,
            128,
            2,
            2,
            conv_kernel_size=4,
            output_gate_activation=variant.removeprefix("gdn-"),
        )
    block = BlockConfig(
        256,
        (
            StageConfig((mixer,)),
            StageConfig((SwiGLUSpec(768),)),
        ),
    )
    return ArchitectureConfig(
        blocks=(BlockGroup(block, repeat=2),),
        embedding_size=256,
        vocab_size=vocab_size,
        max_position_embeddings=sequence_length,
    )


def task_batch(settings, batch_size, seed):
    common = {
        "batch_size": batch_size,
        "sequence_length": settings["sequence_length"],
        "vocab_size": settings["vocab_size"],
        "seed": seed,
    }
    if settings["task"] == "mqar":
        return mqar_batch(**common, num_pairs=settings["num_pairs"])
    if settings["task"] == "palindrome":
        return palindrome_batch(**common)
    return stack_batch(**common, num_stacks=settings["num_stacks"])


def command_output(command):
    result = subprocess.run(command, capture_output=True, check=False, text=True)
    return result.stdout.strip() or None


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def cosine_scale(completed_steps, max_steps):
    if not 0 <= completed_steps <= max_steps:
        raise ValueError("cosine schedule step is outside the training horizon")
    return 0.5 * (1 + math.cos(math.pi * completed_steps / max_steps))


def set_learning_rate(optimizer, learning_rate):
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


@torch.inference_mode()
def evaluate(model, settings, device):
    model.eval()
    correct = 0
    targets_seen = 0
    loss_sum = 0.0
    batches = settings["eval_examples"] // settings["eval_batch_size"]
    for index in range(batches):
        inputs, targets = task_batch(
            settings,
            settings["eval_batch_size"],
            settings["validation_seed"] + index,
        )
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        mask = targets != IGNORE_INDEX
        selected_logits = logits[mask]
        selected_targets = targets[mask]
        loss_sum += F.cross_entropy(
            selected_logits,
            selected_targets,
            reduction="sum",
        ).item()
        correct += (selected_logits.argmax(dim=-1) == selected_targets).sum().item()
        targets_seen += selected_targets.numel()
    return {
        "accuracy": correct / targets_seen,
        "correct": correct,
        "loss": loss_sum / targets_seen,
        "targets": targets_seen,
    }


def base_report(settings, model, source):
    return {
        "format": "speck_synthetic_memory_training",
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "settings": settings,
        "architecture": model.config.settings(),
        "parameters": model.parameter_count(),
        "flops_per_token": model.flops_per_token(settings["sequence_length"]),
        "source": source,
        "history": [],
    }


def run(args):
    settings = resolved_settings(args)
    device = torch.device(settings["device"])
    if settings["variant"] == "kda-sigmoid" and device.type == "cuda":
        try:
            import fla.ops.kda  # noqa: F401
        except ImportError as error:
            raise RuntimeError("CUDA KDA training requires the pinned linear extra") from error
    source = {
        "git_revision": command_output(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(command_output(["git", "status", "--porcelain"])),
    }
    torch.manual_seed(settings["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(settings["seed"])
    architecture = architecture_for_variant(
        settings["variant"],
        settings["sequence_length"],
        settings["vocab_size"],
    )
    model = SpeckForCausalLM(architecture).to(device)
    model.init_weights()
    optimizer = model.optimizer(
        lr=settings["lr"],
        weight_decay=settings["weight_decay"],
        name="adamw",
    )
    train_model = (
        torch.compile(
            model,
            dynamic=False,
            options={
                "max_autotune": True,
                "coordinate_descent_tuning": True,
                "aggressive_fusion": True,
            },
        )
        if settings["compiled"]
        else model
    )
    report = base_report(settings, model, source)
    report["environment"] = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    output = args.output.expanduser().resolve()
    started = time.perf_counter()
    training_seconds = 0.0
    completed = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    initial = evaluate(model, settings, device)
    report["history"].append({"step": 0, **initial})
    atomic_json(output, report)
    print(f"step 0 | val loss {initial['loss']:.5f} | accuracy {initial['accuracy']:.4f}")

    for step in range(settings["max_steps"]):
        model.train()
        set_learning_rate(
            optimizer,
            settings["lr"] * cosine_scale(step, settings["max_steps"]),
        )
        inputs, targets = task_batch(
            settings,
            settings["batch_size"],
            settings["seed"] * 1_000_000 + step,
        )
        inputs = inputs.to(device)
        targets = targets.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = train_model(inputs, targets)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"])
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        training_seconds += time.perf_counter() - train_started
        completed = step + 1
        if completed == 1 or completed % settings["log_every"] == 0:
            print(
                f"step {completed}/{settings['max_steps']} | loss {loss.item():.5f} | "
                f"grad {float(grad_norm):.3f}"
            )
        should_evaluate = (
            completed % settings["eval_every"] == 0
            or completed == settings["max_steps"]
        )
        if should_evaluate:
            validation = evaluate(model, settings, device)
            report["history"].append({"step": completed, **validation})
            report["completed_steps"] = completed
            report["training_seconds"] = training_seconds
            report["tokens_per_training_second"] = (
                completed
                * settings["batch_size"]
                * settings["sequence_length"]
                / training_seconds
            )
            atomic_json(output, report)
            print(
                f"step {completed} | val loss {validation['loss']:.5f} | "
                f"accuracy {validation['accuracy']:.4f}"
            )
            if validation["accuracy"] >= settings["early_stop_accuracy"]:
                report["early_stopped"] = True
                break

    report.update(
        status="complete",
        completed_steps=completed,
        early_stopped=report.get("early_stopped", False),
        training_seconds=training_seconds,
        wall_seconds=time.perf_counter() - started,
        tokens_per_training_second=(
            completed
            * settings["batch_size"]
            * settings["sequence_length"]
            / training_seconds
        ),
        best_validation_accuracy=max(point["accuracy"] for point in report["history"]),
        best_validation_loss=min(point["loss"] for point in report["history"]),
        final_validation=report["history"][-1],
        peak_allocated_bytes=(
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    )
    atomic_json(output, report)
    return report


def main(argv=None):
    report = run(arguments(argv))
    print(
        f"complete: {report['completed_steps']} steps | "
        f"best accuracy {report['best_validation_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
