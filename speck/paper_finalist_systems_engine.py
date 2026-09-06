"""Run one non-persisting finalist systems optimizer-step benchmark."""

import copy
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.base_train import BaseTrainer
from speck.common import cleanup
from speck.config import load_experiment
from speck.dataloader import packed_loader
from speck.paper_finalist_systems_workload import build_trial_plan
from speck.train import optimization_step


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path):
    path = Path(path).expanduser().resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return path, value


def terminal_learning_rate(train):
    lr = train.get("lr")
    minimum = train.get("min_lr")
    if (
        train.get("lr_schedule") != "cosine"
        or isinstance(lr, bool)
        or not isinstance(lr, (int, float))
        or not math.isfinite(lr)
        or lr <= 0
        or isinstance(minimum, bool)
        or not isinstance(minimum, (int, float))
        or not math.isfinite(minimum)
        or not 0 <= minimum <= 1
    ):
        raise ValueError("systems engine cannot derive the frozen terminal learning rate")
    return float(lr * minimum)


def benchmark_configs(trial, protocol):
    configs = load_experiment(
        trial["experiment"],
        "data",
        "tokenizer",
        "model",
        "runtime",
        "train",
    )
    configs = copy.deepcopy(configs)
    train = configs["train"]
    runtime = configs["runtime"]
    workload = protocol["trial_workload"]
    hardware = protocol["hardware_and_software_lock"]
    if (
        train.get("run") != trial["run"]
        or runtime.get("device_batch_size") != hardware["device_batch_size"]
        or train.get("sequence_length") != hardware["sequence_length"]
        or train.get("batch_tokens") != hardware["batch_tokens"]
        or trial["checkpoint_step"] != workload["checkpoint_step"]
        or trial["warmup_optimizer_steps"] != workload["warmup_optimizer_steps"]
        or trial["measured_optimizer_steps"] != workload["measured_optimizer_steps"]
    ):
        raise ValueError("systems engine trial or experiment drifted from the protocol")
    train.update(
        {
            "run": "dummy",
            "train_tokens": workload["total_tokens_per_trial"],
            "eval_every": 0,
            "save_every": 0,
            "checkpoint_tokens": [],
        }
    )
    return configs


def trainer_cli(trial, ephemeral_output, *, device="cuda", no_compile=False):
    return SimpleNamespace(
        experiment=trial["experiment"],
        device=device,
        resume=None,
        no_compile=no_compile,
        branch_from=Path(trial["checkpoint_directory"]),
        branch_step=trial["checkpoint_step"],
        branch_schedule="new",
        branch_kind="same",
        device_batch_size=None,
        save_every=0,
        eval_every=0,
        stop_at_tokens=None,
        output_dir=Path(ephemeral_output),
    )


def execute_windows(
    step,
    synchronize,
    reset_peak,
    marker,
    *,
    warmup_steps=10,
    measured_steps=30,
    clock=time.perf_counter,
    monotonic_ns=time.monotonic_ns,
):
    if warmup_steps != 10 or measured_steps != 30:
        raise ValueError("systems engine windows changed from the frozen protocol")
    marker("warmup_start", monotonic_ns())
    for _ in range(warmup_steps):
        step()
    synchronize()
    reset_peak()
    boundary = monotonic_ns()
    marker("warmup_end", boundary)
    marker("measured_start", boundary)
    started = clock()
    for _ in range(measured_steps):
        step()
    synchronize()
    measured_seconds = clock() - started
    marker("measured_end", monotonic_ns())
    if not math.isfinite(measured_seconds) or measured_seconds <= 0:
        raise ValueError("systems engine measured interval is invalid")
    return measured_seconds


class DeviceBatchReplay:
    def __init__(self, batches, device):
        self.batches = batches
        self.device = device
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.batches):
            raise StopIteration
        inputs, targets, state = self.batches[self.index]
        self.index += 1
        return inputs.to(self.device), targets.to(self.device), copy.deepcopy(state)


def _batch_fingerprint(batches):
    digest = hashlib.sha256()
    for inputs, targets, state in batches:
        for tensor in (inputs, targets):
            tensor = tensor.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(json.dumps(list(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
        digest.update(json.dumps(state, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def materialize_paired_batches(trainer, resume_state, total_steps, *, loader_factory=packed_loader):
    if total_steps != 40 or trainer.accumulation < 1:
        raise ValueError("systems engine batch materialization changed from the frozen workload")
    loader = loader_factory(
        trainer.tokenizer,
        trainer.args.device_batch_size,
        trainer.args.sequence_length,
        "train",
        device="cpu",
        resume_state_dict=resume_state,
        data_dir=trainer.args.data_dir,
    )
    batch_count = 1 + total_steps * trainer.accumulation
    batches = [next(loader) for _ in range(batch_count)]
    if any(
        inputs.device.type != "cpu" or targets.device.type != "cpu"
        for inputs, targets, _ in batches
    ):
        raise ValueError("systems engine paired batches were not materialized on CPU")
    fingerprint = _batch_fingerprint(batches)
    replay = DeviceBatchReplay(batches, trainer.device)
    trainer.inputs, trainer.targets, trainer.data_state = next(replay)
    trainer.train_data = replay
    return {
        "sha256": fingerprint,
        "microbatches": batch_count,
        "optimizer_steps": total_steps,
        "accumulation_steps": trainer.accumulation,
        "materialized_device": "cpu",
        "transfer_device": trainer.device.type,
        "H2D_inside_optimizer_windows": True,
    }


def run_engine(
    trial,
    protocol,
    ephemeral_output,
    *,
    trainer_factory=BaseTrainer,
    optimization_step_fn=optimization_step,
    batch_materializer=materialize_paired_batches,
    cleanup_fn=cleanup,
    device="cuda",
    no_compile=False,
    clock=time.perf_counter,
    monotonic_ns=time.monotonic_ns,
):
    """Reuse frozen initialization/restore/step mechanics without trainer output paths."""

    configs = benchmark_configs(trial, protocol)
    cli = trainer_cli(trial, ephemeral_output, device=device, no_compile=no_compile)
    markers = {}

    def mark(name, timestamp):
        if name in markers:
            raise ValueError(f"systems engine phase marker repeated: {name}")
        markers[name] = timestamp

    mark("process_start", monotonic_ns())
    trainer = None
    try:
        trainer = trainer_factory(configs, cli)
        trainer._initialize_runtime()
        trainer._load_and_verify_data()
        trainer._initialize_model_and_geometry()
        trainer._restore_training_state()
        restored_data_state = copy.deepcopy(trainer.data_state)
        mark("checkpoint_loaded", monotonic_ns())
        trainer._build_resolved_settings()
        trainer._prepare_execution()
        batches = batch_materializer(
            trainer,
            restored_data_state,
            trial["warmup_optimizer_steps"] + trial["measured_optimizer_steps"],
        )
        mark("batches_materialized", monotonic_ns())
        terminal_lr = terminal_learning_rate(configs["train"])

        def step():
            loss, _, batch = optimization_step_fn(
                trainer.train_model,
                trainer.parameters,
                trainer.optimizer,
                trainer.train_data,
                (trainer.inputs, trainer.targets, trainer.data_state),
                trainer.accumulation,
                trainer.args.grad_clip,
                terminal_lr,
                trainer.distributed,
            )
            trainer.inputs, trainer.targets, trainer.data_state = batch
            return loss

        def synchronize():
            if trainer.device.type == "cuda":
                torch.cuda.synchronize()

        def reset_peak():
            if trainer.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(trainer.device)

        measured_seconds = execute_windows(
            step,
            synchronize,
            reset_peak,
            mark,
            warmup_steps=trial["warmup_optimizer_steps"],
            measured_steps=trial["measured_optimizer_steps"],
            clock=clock,
            monotonic_ns=monotonic_ns,
        )
        peak_allocated = (
            torch.cuda.max_memory_allocated(trainer.device)
            if trainer.device.type == "cuda"
            else None
        )
        peak_reserved = (
            torch.cuda.max_memory_reserved(trainer.device)
            if trainer.device.type == "cuda"
            else None
        )
        return {
            "status": "complete_unintegrated",
            "arm_id": trial["arm_id"],
            "run": trial["run"],
            "checkpoint_step": trial["checkpoint_step"],
            "warmup_optimizer_steps": trial["warmup_optimizer_steps"],
            "measured_optimizer_steps": trial["measured_optimizer_steps"],
            "measured_tokens": trial["measured_tokens"],
            "measured_wall_seconds": measured_seconds,
            "terminal_learning_rate": terminal_lr,
            "paired_batches": batches,
            "phase_markers": markers,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "non_finite_steps": 0,
            "OOM": False,
            "kernel_fallback": None,
            "validation_executed": False,
            "checkpoint_write_executed": False,
            "summary_write_executed": False,
            "tracking_initialized": False,
            "model_or_optimizer_output_persisted": False,
        }
    finally:
        cleanup_fn()
        mark("process_end", monotonic_ns())


def validate_activation(path, protocol_path):
    path, activation = load_object(path)
    if (
        activation.get("format") != "speck_paper_finalist_systems_activation"
        or activation.get("format_version") != 1
        or activation.get("status") != "qualified_post_language_systems_execution"
        or activation.get("protocol_sha256") != file_sha256(protocol_path)
        or activation.get("engine_sha256") != file_sha256(__file__)
        or activation.get("accepted_results") != 12
        or activation.get("v3_acceptance_pass") is not True
        or activation.get("actual_path_GPU_preflight_pass") is not True
    ):
        raise ValueError("systems engine activation is absent or invalid")
    return path, activation


def run_trial(
    protocol_path,
    qualification_path,
    activation_path,
    block,
    position,
    output_path,
):
    """Run only after a future post-language activation artifact exists."""

    protocol_path, protocol = load_object(protocol_path)
    validate_activation(activation_path, protocol_path)
    from scripts.paper_finalist_result_acceptance_validate_v3 import validate_program

    acceptance = validate_program()
    if acceptance["accepted_results"] != 12 or not acceptance["analysis_complete"]:
        raise ValueError("systems engine requires complete v3-accepted language evidence")
    output_path = Path(output_path).expanduser().resolve()
    plan = build_trial_plan(protocol_path, qualification_path, output_path.parent)
    matches = [
        trial
        for trial in plan["trials"]
        if trial["block"] == block and trial["position"] == position
    ]
    if len(matches) != 1 or Path(matches[0]["output"]) != output_path:
        raise ValueError("systems engine output does not match the exact trial plan")
    return run_engine(matches[0], protocol, output_path.with_suffix(".ephemeral"))
