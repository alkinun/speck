"""shared training-step mechanics."""

from contextlib import nullcontext
import math

import torch
import torch.distributed as dist


def lr_scale(step, steps, warmup, minimum):
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, steps - warmup))
    return minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * progress))


def sequence_length_for_step(step, steps, final_length, enabled):
    if not enabled:
        return final_length
    if step * 4 < steps:
        return final_length // 2
    return final_length


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
        context = train_model.no_sync() if distributed and micro_step + 1 < accumulation else nullcontext()
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
