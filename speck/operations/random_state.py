"""Persist process-local generators, including each distributed training rank."""

import base64
import random

import numpy as np
import torch
import torch.distributed as dist


def seed_generators(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def capture_rng(device):
    name, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        "python": random.getstate(),
        "numpy": (name, keys.tolist(), position, has_gauss, cached_gaussian),
    }


def restore_rng(value, device):
    if (value["torch_cuda"] is not None) != (device.type == "cuda"):
        raise ValueError("checkpoint RNG device differs")
    torch.set_rng_state(value["torch_cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(value["torch_cuda"], device)
    random.setstate(value["python"])
    name, keys, position, has_gauss, cached_gaussian = value["numpy"]
    np.random.set_state(
        (name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached_gaussian)
    )


def _json_state(value):
    return {
        **value,
        **{
            name: base64.b64encode(bytes(value[name].tolist())).decode()
            if value[name] is not None
            else None
            for name in ("torch_cpu", "torch_cuda")
        },
    }


def gather_training_rng(device, world_size):
    """Gather compact JSON payloads into the existing atomic metadata publication."""
    local = capture_rng(device)
    try:
        states = [_json_state(local)]
        if world_size > 1:
            states = [None] * world_size
            dist.all_gather_object(states, _json_state(local))
        return {
            "format": "speck_training_rng",
            "format_version": 1,
            "world_size": world_size,
            "ranks": states,
        }
    finally:
        restore_rng(local, device)


def restore_training_rng(value, device, rank, world_size):
    """Old checkpoints remain readable, without claiming full generator continuity."""
    if value is None:
        return False
    if (
        value.get("format") != "speck_training_rng"
        or value.get("format_version") != 1
        or value.get("world_size") != world_size
        or len(value.get("ranks", [])) != world_size
        or not 0 <= rank < world_size
    ):
        raise ValueError("checkpoint RNG rank/world-size contract differs")
    try:
        state = dict(value["ranks"][rank])
        for name in ("torch_cpu", "torch_cuda"):
            if state[name] is not None:
                state[name] = torch.tensor(
                    list(base64.b64decode(state[name], validate=True)), dtype=torch.uint8
                )
        version, keys, gaussian = state["python"]
        state["python"] = (version, tuple(keys), gaussian)
        # Validate before changing any process-global generators.
        random.Random().setstate(state["python"])
        np.random.RandomState().set_state(
            (state["numpy"][0], np.asarray(state["numpy"][1], dtype=np.uint32), *state["numpy"][2:])
        )
        torch.Generator().set_state(state["torch_cpu"])
        if (state["torch_cuda"] is not None) != (device.type == "cuda"):
            raise ValueError("checkpoint RNG device differs")
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError("invalid checkpoint RNG state") from error
    restore_rng(state, device)
    return True


def warmup_resume_backend(model, device, shapes, vocab_size, *, loss_reduction=None):
    """Initialize lazy forward/backward kernels before loading saved tensors and RNG.

    No optimizer step or loader advancement occurs. Eager CUDA restart is the initial
    qualification target; compilation and distributed execution need their own rehearsal.
    """
    for batch_size, length in shapes:
        tokens = torch.arange(batch_size * length + 1, device=device) % vocab_size
        inputs, targets = (
            tokens[:-1].reshape(batch_size, length),
            tokens[1:].reshape(batch_size, length),
        )
        options = {}
        if loss_reduction is not None:
            options["loss_reduction"] = loss_reduction
            targets = targets.clone()
            targets[:, 1::2] = -100
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            loss = model(inputs, targets, **options)
            loss.backward()
    model.zero_grad(set_to_none=True)
