"""Provide shared training-step mechanics."""

import math
from contextlib import nullcontext

import torch
import torch.distributed as dist


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


def lr_scale(step, steps, warmup, minimum):
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


class UpdateMonitor:
    """Measure exact relative updates for a small fixed parameter sample."""

    def __init__(self, named_parameters):
        self.parameters = tuple(named_parameters)

    @property
    def names(self):
        return tuple(name for name, _ in self.parameters)

    @torch.no_grad()
    def snapshot(self):
        return tuple(parameter.detach().clone() for _, parameter in self.parameters)

    @torch.no_grad()
    def metrics(self, snapshot):
        values = {}
        for (name, parameter), before in zip(self.parameters, snapshot):
            weight_norm = before.norm()
            update_norm = (parameter - before).norm()
            values[name] = {
                "weight_norm": weight_norm.item(),
                "update_norm": update_norm.item(),
                "effective_lr": (
                    update_norm / weight_norm.clamp_min(torch.finfo(before.dtype).tiny)
                ).item(),
            }
        return values


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
):
    optimizer.zero_grad(set_to_none=True)
    loss_sum = torch.zeros((), device=batch[0].device)
    if cudagraphs:
        torch.compiler.cudagraph_mark_step_begin()
    for micro_step in range(accumulation):
        context = (
            train_model.no_sync()
            if distributed and micro_step + 1 < accumulation
            else nullcontext()
        )
        with context:
            loss = train_model(batch[0], batch[1])
            (loss / accumulation).backward()
        loss_sum += loss.detach()
        batch = next(loader)

    finite = torch.isfinite(loss_sum).to(torch.int32)
    if distributed:
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not finite.item():
        raise FloatingPointError("non-finite training loss")
    for group in optimizer.param_groups:
        group["lr"] = lr
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, grad_clip, error_if_nonfinite=True)
    optimizer.step()
    return loss_sum / accumulation, grad_norm, batch
