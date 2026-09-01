"""Provide shared training-step mechanics."""

import math
from contextlib import nullcontext

import torch
import torch.distributed as dist

from speck.model import CausalLMTrainingOutput, RoutingLayerStats


def assert_finite(value, message, distributed=False):
    """Check a scalar without introducing a CUDA host synchronization."""

    finite = torch.isfinite(value).to(torch.int32)
    if distributed:
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if value.device.type == "cuda":
        torch._assert_async(finite, message)
    elif not finite.item():
        raise FloatingPointError(message)


def set_optimizer_lr(optimizer, lr):
    """Update scalar tensor learning rates in place to keep compiled steps reusable."""

    for group in optimizer.param_groups:
        current = group["lr"]
        if isinstance(current, torch.Tensor):
            current.fill_(lr)
        else:
            group["lr"] = lr


def resolve_device_batch_size(limit, batch_tokens, sequence_length, world_size):
    values = (limit, batch_tokens, sequence_length, world_size)
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in values):
        raise ValueError("batch geometry values must be positive integers")
    distributed_sequence_tokens = sequence_length * world_size
    if batch_tokens % distributed_sequence_tokens:
        raise ValueError("batch tokens must contain a whole sequence on every rank")
    device_batch_size = min(limit, batch_tokens // distributed_sequence_tokens)
    if batch_tokens % (device_batch_size * distributed_sequence_tokens):
        raise ValueError("batch tokens must be divisible by the distributed microbatch")
    return device_batch_size


def validate_loader_progress(data_state, trained_tokens):
    """Require a next-batch loader cursor at the checkpoint's trained-token boundary."""

    if not isinstance(data_state, dict):
        raise ValueError("checkpoint is missing packed loader state")
    offset = data_state.get("global_consumed_tokens")
    if offset != trained_tokens:
        raise ValueError(
            "checkpoint loader offset does not match training progress: "
            f"expected {trained_tokens:,}, got {offset!r}"
        )


def lr_scale(step, steps, warmup, minimum, schedule="cosine"):
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not isinstance(step, int) or not 0 <= step < steps:
        raise ValueError("step must be an executed zero-based schedule index")
    if not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    if not isinstance(minimum, (int, float)) or not math.isfinite(minimum):
        raise ValueError("minimum must be a finite scale")
    if not 0 <= minimum <= 1:
        raise ValueError("minimum must be between zero and one")
    if schedule not in {"constant", "cosine"}:
        raise ValueError("unsupported learning-rate schedule")
    if schedule == "constant" and (warmup != 0 or minimum != 1):
        raise ValueError("constant schedule requires zero warmup and minimum scale one")

    if schedule == "constant":
        return 1.0
    if step < warmup:
        return (step + 1) / warmup
    if steps == 1:
        return minimum
    if warmup == 0:
        progress = step / (steps - 1)
    else:
        progress = (step - warmup + 1) / (steps - warmup)
    return minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * progress))


def checkpoint_milestones(tokens, batch_tokens, global_token_offset, steps):
    """Map requested global-token milestones to phase-local completed steps."""

    if not isinstance(tokens, list) or any(
        isinstance(token, bool) or not isinstance(token, int) or token < 1 for token in tokens
    ):
        raise ValueError("checkpoint_tokens must be a list of positive integers")
    if tokens != sorted(set(tokens)):
        raise ValueError("checkpoint_tokens must be sorted and unique")
    if (
        isinstance(batch_tokens, bool)
        or not isinstance(batch_tokens, int)
        or batch_tokens < 1
        or isinstance(global_token_offset, bool)
        or not isinstance(global_token_offset, int)
        or global_token_offset < 0
        or global_token_offset % batch_tokens
    ):
        raise ValueError("global token geometry is invalid")
    milestones = {}
    for token in tokens:
        if token <= global_token_offset:
            continue
        step = math.ceil((token - global_token_offset) / batch_tokens)
        if step <= steps:
            milestones[step] = token
    return milestones


def checkpoint_global_tokens(metadata, batch_tokens):
    global_tokens = metadata.get("global_tokens")
    if global_tokens is None:
        global_tokens = metadata.get("global_step", metadata["step"]) * batch_tokens
    if global_tokens % batch_tokens:
        raise ValueError("checkpoint global tokens are not aligned to the optimizer batch")
    return global_tokens


def branch_position(metadata, batch_tokens, steps=None):
    resolved = metadata["resolved"]
    global_token_offset = checkpoint_global_tokens(metadata, batch_tokens)
    data_token_offset = metadata["data_state"]["global_consumed_tokens"]
    if global_token_offset % batch_tokens or data_token_offset % batch_tokens:
        raise ValueError("branch parent is not aligned to the optimizer batch")
    global_step = global_token_offset // batch_tokens
    if metadata.get("global_step", metadata["step"]) != global_step:
        raise ValueError("branch parent global step does not match its data position")
    schedule_step_offset = resolved.get("schedule_step_offset", 0) + metadata["step"]
    schedule_steps = resolved.get("schedule_steps", resolved["steps"])
    if steps is not None and schedule_step_offset + steps > schedule_steps:
        raise ValueError("branch exceeds the parent learning-rate schedule")
    return global_token_offset, data_token_offset, schedule_step_offset, schedule_steps


def optimization_step(
    train_model,
    parameters,
    optimizer,
    loader,
    batch,
    accumulation,
    grad_clip,
    lr,
    distributed=False,
    cudagraphs=False,
    return_training_output=False,
    load_balance_coefficient=0.01,
    router_z_loss_coefficient=0.001,
):
    optimizer.zero_grad(set_to_none=True)
    loss_sum = torch.zeros((), device=batch[0].device)
    lm_loss_sum = torch.zeros_like(loss_sum)
    load_balance_sum = torch.zeros_like(loss_sum)
    z_loss_sum = torch.zeros_like(loss_sum)
    routing_sums = None
    if cudagraphs:
        torch.compiler.cudagraph_mark_step_begin()
    for micro_step in range(accumulation):
        context = (
            train_model.no_sync()
            if distributed and micro_step + 1 < accumulation
            else nullcontext()
        )
        with context:
            output = train_model(
                batch[0],
                batch[1],
                return_training_output=return_training_output,
                load_balance_coefficient=load_balance_coefficient,
                router_z_loss_coefficient=router_z_loss_coefficient,
            )
            loss = output.total_loss if return_training_output else output
            (loss / accumulation).backward()
        loss_sum += loss.detach()
        if return_training_output:
            if not isinstance(output, CausalLMTrainingOutput):
                raise TypeError("model did not return a typed training output")
            lm_loss_sum += output.lm_loss.detach()
            load_balance_sum += output.load_balance_loss.detach()
            z_loss_sum += output.z_loss.detach()
            if routing_sums is None:
                routing_sums = [
                    {
                        "stats": stats,
                        "mean_probabilities": stats.mean_probabilities.detach().clone(),
                        "utilization": stats.utilization.detach().clone(),
                        "entropy": stats.entropy.detach().clone(),
                        "load_balance_loss": stats.load_balance_loss.detach().clone(),
                        "z_loss": stats.z_loss.detach().clone(),
                    }
                    for stats in output.routing
                ]
            else:
                if len(routing_sums) != len(output.routing):
                    raise RuntimeError("routed layer count changed during accumulation")
                for accumulated, stats in zip(routing_sums, output.routing):
                    if accumulated["stats"].layer != stats.layer:
                        raise RuntimeError("routed layer order changed during accumulation")
                    for name in (
                        "mean_probabilities",
                        "utilization",
                        "entropy",
                        "load_balance_loss",
                        "z_loss",
                    ):
                        accumulated[name].add_(getattr(stats, name).detach())
        batch = next(loader)

    assert_finite(loss_sum, "non-finite training loss", distributed)
    set_optimizer_lr(optimizer, lr)
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
    assert_finite(grad_norm, "non-finite training gradients")
    optimizer.step()
    if not return_training_output:
        return loss_sum / accumulation, grad_norm, batch
    routing = tuple(
        RoutingLayerStats(
            layer=accumulated["stats"].layer,
            mean_probabilities=accumulated["mean_probabilities"] / accumulation,
            utilization=accumulated["utilization"] / accumulation,
            entropy=accumulated["entropy"] / accumulation,
            load_balance_loss=accumulated["load_balance_loss"] / accumulation,
            z_loss=accumulated["z_loss"] / accumulation,
        )
        for accumulated in (routing_sums or [])
    )
    return (
        CausalLMTrainingOutput(
            total_loss=loss_sum / accumulation,
            lm_loss=lm_loss_sum / accumulation,
            load_balance_loss=load_balance_sum / accumulation,
            z_loss=z_loss_sum / accumulation,
            routing=routing,
        ),
        grad_norm,
        batch,
    )


def average_training_output(output, distributed):
    """Average detached typed loss and routing diagnostics across ranks in place."""

    if not distributed:
        return output
    tensors = [
        output.total_loss,
        output.lm_loss,
        output.load_balance_loss,
        output.z_loss,
    ]
    for stats in output.routing:
        tensors.extend(
            (
                stats.mean_probabilities,
                stats.utilization,
                stats.entropy,
                stats.load_balance_loss,
                stats.z_loss,
            )
        )
    for tensor in tensors:
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    return output
