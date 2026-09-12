"""Qualify checkpointed training mechanics for corrected tokenizer-pilot runs."""

from pathlib import Path

import torch

from speck.checkpoint import load, save
from speck.model import build_model
from speck.tokenizer_pilot_runs import _fingerprint
from speck.tokenizer_pilot_runtime import (
    PilotBatchLoader,
    PilotTokenStream,
    validate_pilot_run_manifest,
)
from speck.train import lr_scale, optimization_step
from speck.validation import positive_integer


def training_boundaries(run):
    """Return mandatory curve, checkpoint, fixed-document, and final boundaries."""

    run = validate_pilot_run_manifest(run)
    final = run["stops"]["final_step"]
    curve_every = run["settings"]["learning_curve_every_steps"]
    checkpoint_every = run["settings"]["checkpoint_every_steps"]
    curve = set(range(curve_every, final, curve_every))
    checkpoints = set(range(checkpoint_every, final, checkpoint_every))
    mandatory = {run["stops"]["fixed_document_step"], final}
    return {
        "learning_curve_steps": tuple(sorted(curve | mandatory)),
        "checkpoint_steps": tuple(sorted(checkpoints | mandatory)),
        "fixed_document_step": run["stops"]["fixed_document_step"],
        "final_step": final,
    }


def learning_rate_for_step(run, step):
    """Resolve the exact zero-based learning rate for one optimizer boundary."""

    run = validate_pilot_run_manifest(run)
    settings = run["settings"]
    return settings["learning_rate"] * lr_scale(
        step,
        run["stops"]["final_step"],
        settings["warmup_steps"],
        settings["min_lr"],
        settings["lr_schedule"],
    )


def build_pilot_model(run, device):
    """Construct and verify the run's exact model before initialization."""

    run = validate_pilot_run_manifest(run)
    device = torch.device(device)
    tokenizer = run["tokenizer"]
    model = build_model(
        run["model"]["settings"],
        tokenizer["vocab_size"],
        tokenizer["bos_token_id"],
        tokenizer["eos_token_id"],
        loss_backend=run["settings"]["loss_backend"],
    )
    if (
        _fingerprint(model.config.settings()) != run["model"]["sha256"]
        or model.parameter_count() != run["model"]["parameters"]
        or model.flops_per_token(run["settings"]["sequence_length"])
        != run["model"]["analytic_training_flops_per_token"]
    ):
        raise ValueError("constructed tokenizer pilot model differs from its run manifest")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    return model.to(device=device, dtype=dtype)


def _state_equal(left, right):
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _state_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _state_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _execute_step(run, model, optimizer, loader, batch, step):
    parameters = tuple(model.parameters())
    return optimization_step(
        model,
        parameters,
        optimizer,
        loader,
        batch,
        run["settings"]["accumulation"],
        run["settings"]["grad_clip"],
        learning_rate_for_step(run, step),
    )


def qualify_checkpoint_resume(run, output_directory, *, device="cpu", steps=2):
    """Compare uninterrupted and restored execution without granting screen authority."""

    run = validate_pilot_run_manifest(run)
    steps = positive_integer(steps, "qualification steps")
    if steps != 2:
        raise ValueError("tokenizer pilot checkpoint qualification requires exactly two steps")
    if run["authority"]["screen_execution"] is not False:
        raise ValueError("qualification requires an execution-blocked run")
    output_directory = Path(output_directory)
    if output_directory.exists():
        raise FileExistsError(f"tokenizer pilot training qualification exists: {output_directory}")
    checkpoint_directory = output_directory / "checkpoints"
    stream = PilotTokenStream(run, verify_hashes=False)
    torch.manual_seed(run["seed"])
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(run["seed"])
    model = build_pilot_model(run, device)
    model.init_weights()
    optimizer = model.optimizer(
        run["settings"]["learning_rate"],
        run["settings"]["weight_decay"],
        run["settings"]["optimizer"],
    )
    loader = PilotBatchLoader(stream, device=device)
    batch = next(loader)
    first_loss, _, batch = _execute_step(run, model, optimizer, loader, batch, 0)
    metadata = {
        "step": 1,
        "run_fingerprint": run["run_fingerprint"],
        "data_state": batch[2],
        "qualification": True,
        "screen_execution": False,
    }
    save(
        checkpoint_directory,
        1,
        model.state_dict(),
        optimizer.state_dict(),
        metadata,
    )
    second_loss, _, continued_batch = _execute_step(run, model, optimizer, loader, batch, 1)
    continued_model = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    continued_optimizer = optimizer.state_dict()

    restored_model = build_pilot_model(run, device)
    restored_optimizer = restored_model.optimizer(
        run["settings"]["learning_rate"],
        run["settings"]["weight_decay"],
        run["settings"]["optimizer"],
    )
    model_state, optimizer_state, loaded_metadata = load(checkpoint_directory, 1, device)
    if loaded_metadata != metadata:
        raise ValueError("tokenizer pilot qualification checkpoint metadata changed")
    restored_model.load_state_dict(model_state)
    restored_optimizer.load_state_dict(optimizer_state)
    restored_loader = PilotBatchLoader(stream, device=device, resume_state=metadata["data_state"])
    restored_batch = next(restored_loader)
    replay_loss, _, replay_batch = _execute_step(
        run,
        restored_model,
        restored_optimizer,
        restored_loader,
        restored_batch,
        1,
    )
    replay_model = {key: value.detach().cpu() for key, value in restored_model.state_dict().items()}
    replay_optimizer = restored_optimizer.state_dict()
    loss_equal = torch.equal(second_loss.detach().cpu(), replay_loss.detach().cpu())
    model_equal = _state_equal(continued_model, replay_model)
    optimizer_equal = _state_equal(continued_optimizer, replay_optimizer)
    cursor_equal = continued_batch[2] == replay_batch[2]
    if not (loss_equal and model_equal and optimizer_equal and cursor_equal):
        raise RuntimeError("tokenizer pilot checkpoint replay is not exactly equivalent")
    return {
        "format": "speck_tokenizer_pilot_training_qualification",
        "format_version": 1,
        "status": "two_step_checkpoint_resume_exact_no_screen_authority",
        "run_id": run["run_id"],
        "run_fingerprint": run["run_fingerprint"],
        "device": str(torch.device(device)),
        "dtype": str(next(model.parameters()).dtype),
        "first_loss": float(first_loss),
        "second_loss": float(second_loss),
        "replay_loss": float(replay_loss),
        "checkpoint_step": 1,
        "next_data_state": replay_batch[2],
        "equivalence": {
            "loss": loss_equal,
            "model": model_equal,
            "optimizer": optimizer_equal,
            "data_cursor": cursor_equal,
        },
        "qualification_optimizer_boundaries_executed": 3,
        "scientific_model_outputs_created": 0,
        "scientific_run": False,
        "screen_execution_authority": False,
        "D5_opening_authority": False,
    }
