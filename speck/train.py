"""Provide shared training-step mechanics."""

import math
from contextlib import nullcontext

import torch
import torch.distributed as dist


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
    if not isinstance(warmup, int) or not 0 <= warmup < steps:
        raise ValueError("warmup must leave at least one scheduled step")
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
):
    optimizer.zero_grad(set_to_none=True)
    loss_sum = torch.zeros((), device=batch[0].device)
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
