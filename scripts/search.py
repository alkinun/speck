"""Run resumable Speck architecture searches."""

import argparse
import fcntl
import json
import os
import platform
import resource
import shutil
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch

from speck.architecture import ArchitectureConfig
from speck.checkpoint import latest, load, load_model, save
from speck.common import base_dir
from speck.config import load_experiment
from speck.dataloader import manifest_fingerprint, packed_loader
from speck.dataset import default_data_dir, load_manifest, verify_shards
from speck.model import SpeckForCausalLM
from speck.search import (
    StudyStore,
    aggregate_final_runs,
    architecture_metrics,
    atomic_json,
    initial_generation,
    later_generation,
    load_search_settings,
    loader_state,
    materialize_generation,
    normalize_baseline,
    open_study,
    percentile_ranks,
    promotion_for_rung,
    prune_checkpoints,
    retained_checkpoint_candidates,
    score_candidates,
    select_finalists,
    status_snapshot,
    utc_now,
    validation_slices,
)
from speck.tokenizer import get_tokenizer
from speck.train import lr_scale, optimization_step, validate_loader_progress


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _runtime(device_name, seed):
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda is required but unavailable")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_float32_matmul_precision("high")
    return device


def _runtime_contract(settings, device):
    device = torch.device(device)
    expected = settings["profile"]["device"]
    if device.type != expected:
        raise ValueError(f"search profile requires {expected}, not {device.type}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda is required but unavailable")
    return {
        "device": str(device),
        "device_type": device.type,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "parameter_dtype": settings["profile"]["parameter_dtype"],
        "compute_dtype": settings["profile"]["compute_dtype"],
        "deterministic_algorithms": settings["training"]["deterministic"],
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }


def _study_inputs(experiment):
    configs = load_experiment(experiment, "model", "data", "tokenizer")
    tokenizer = get_tokenizer(**configs["tokenizer"])
    data_dir = configs["data"].get("output_dir") or str(default_data_dir / "packed")
    manifest = load_manifest(data_dir)
    provenance = {
        "configs": configs,
        "tokenizer": {
            "fingerprint": tokenizer.fingerprint(),
            "vocab_size": tokenizer.vocab_size,
            "bos_token_id": tokenizer.bos_id,
            "eos_token_id": tokenizer.eos_id,
        },
        "manifest": manifest_fingerprint(manifest),
        "data_dir": str(Path(data_dir).resolve()),
    }
    return configs, provenance


def _verify_inputs(inputs):
    manifest = load_manifest(inputs["data_dir"])
    if manifest_fingerprint(manifest) != inputs["manifest"]:
        raise ValueError("packed dataset manifest changed")
    verify_shards(inputs["data_dir"], manifest)


def _verify_runtime(state, settings, device):
    if state["provenance"]["runtime"] != _runtime_contract(settings, device):
        raise ValueError("study runtime contract changed")


def _cpu_contract(settings):
    return {
        "device": "cpu",
        "device_name": platform.processor() or platform.machine(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "logical_cores": os.cpu_count(),
        "threads": settings["final_profile"]["cpu_threads"],
        "torch": torch.__version__,
    }


def _context(study, candidate_id):
    store = StudyStore(study)
    settings = store.settings()
    state = store.state()
    configs, inputs = _study_inputs(state["experiment"])
    if state["provenance"]["inputs"] != inputs:
        raise ValueError("study comparison inputs changed")
    candidate = store.candidate_path(candidate_id)
    architecture = ArchitectureConfig.from_dict(
        json.loads((candidate / "architecture.json").read_text(encoding="utf-8"))
    )
    return store, settings, state, configs, candidate, architecture


def _model(architecture, device, seed):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    model = SpeckForCausalLM(architecture).to(device)
    model.init_weights()
    return model


def check_candidate(study, candidate_id, device_name):
    store, settings, state, configs, candidate, architecture = _context(study, candidate_id)
    device = _runtime(device_name, settings["seed"])
    _verify_runtime(state, settings, device)
    model = _model(architecture, device, settings["seed"])
    generator = torch.Generator(device=device).manual_seed(settings["seed"])
    tokens = torch.randint(
        architecture.vocab_size,
        (2, 2048),
        device=device,
        generator=generator,
    )
    loss = model(tokens, tokens)
    if not torch.isfinite(loss):
        raise FloatingPointError("model produced a non-finite feasibility loss")
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
        raise FloatingPointError("model produced non-finite feasibility gradients")

    model.zero_grad(set_to_none=True)
    model.eval()
    length = min(16, architecture.max_position_embeddings)
    fixture = torch.randint(
        architecture.vocab_size,
        (1, length),
        device=device,
        generator=generator,
    )
    with torch.inference_mode():
        full = model(fixture)
        sequence_state = model.state(length=length)
        cached = torch.cat(
            [model(fixture[:, index : index + 1], state=sequence_state) for index in range(length)],
            dim=1,
        )
    if not torch.isfinite(full).all() or not torch.isfinite(cached).all():
        raise FloatingPointError("model produced non-finite feasibility outputs")
    tolerance = settings["final_profile"]
    if not torch.allclose(
        full,
        cached,
        atol=tolerance["cache_absolute_tolerance"],
        rtol=tolerance["cache_relative_tolerance"],
    ):
        raise ValueError("full-sequence and cached decoding outputs differ")
    store.update_result(candidate_id, feasibility={"status": "passed"})
    return {"status": "passed"}


def _request(model, fixture, prompt_length, generated_tokens, device, measured):
    durations = {}
    state = model.state(length=prompt_length + generated_tokens)
    if measured:
        synchronize(device)
        started = time.perf_counter()
    model(fixture[:, :prompt_length], state=state, last_token_only=True)
    if measured:
        synchronize(device)
        durations["prefill"] = time.perf_counter() - started
    decode = []
    for index in range(generated_tokens):
        if measured:
            synchronize(device)
            started = time.perf_counter()
        model(
            fixture[:, prompt_length + index : prompt_length + index + 1],
            state=state,
            last_token_only=True,
        )
        if measured:
            synchronize(device)
            decode.append(time.perf_counter() - started)
    if measured:
        durations["first_decode"] = decode[0]
        durations["steady_decode"] = statistics.mean(decode)
    return durations


def _distribution(values):
    return {
        "p50_seconds": percentile(values, 0.50),
        "p95_seconds": percentile(values, 0.95),
    }


def _measure_profile(model, architecture, profile, device):
    model.eval()
    maximum_prompt = max(profile["prompt_lengths"])
    generated_tokens = profile["generated_tokens"]
    if maximum_prompt + generated_tokens > architecture.max_position_embeddings:
        raise ValueError("profile request exceeds the model context")
    fixture = torch.randint(
        architecture.vocab_size,
        (1, maximum_prompt + generated_tokens),
        device=device,
        generator=torch.Generator(device=device).manual_seed(profile["seed"]),
    )

    with torch.inference_mode():
        for _ in range(profile["warmups"]):
            for prompt_length in profile["prompt_lengths"]:
                _request(model, fixture, prompt_length, generated_tokens, device, False)
        synchronize(device)
        resident = torch.cuda.memory_allocated(device) if device.type == "cuda" else None
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        samples = {
            "prefill_512": [],
            "prefill_2048": [],
            "decode_2048": [],
            "steady_decode_64": [],
        }
        for _ in range(profile["requests"]):
            for prompt_length in profile["prompt_lengths"]:
                measured = _request(
                    model,
                    fixture,
                    prompt_length,
                    generated_tokens,
                    device,
                    True,
                )
                samples[f"prefill_{prompt_length}"].append(measured["prefill"])
                if prompt_length == maximum_prompt:
                    samples["decode_2048"].append(measured["first_decode"])
                    samples["steady_decode_64"].append(measured["steady_decode"])
    return {
        "contract": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "parameter_dtype": profile["parameter_dtype"],
            "compute_dtype": profile["compute_dtype"],
            "warmups": profile["warmups"],
            "requests": profile["requests"],
            "seed": profile["seed"],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        },
        "latency": {name: _distribution(values) for name, values in samples.items()},
        "memory": {
            "resident_vram_bytes": resident,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
        },
    }


def profile_candidate(study, candidate_id, device_name):
    store, settings, state, configs, candidate, architecture = _context(study, candidate_id)
    profile = settings["profile"]
    device = _runtime(device_name, profile["seed"])
    _verify_runtime(state, settings, device)
    model = _model(architecture, device, profile["seed"])
    measured = _measure_profile(model, architecture, profile, device)
    result = store.results()
    current = next(value for value in result if value["candidate_id"] == candidate_id)
    candidate_profile = dict(current.get("profile") or {})
    candidate_profile.update(measured)
    store.update_result(candidate_id, profile=candidate_profile)
    return candidate_profile


@torch.no_grad()
def evaluate_nll(model, tokenizer, settings, data_dir, offset, target_tokens, device):
    training = settings["training"]
    batch_tokens = training["device_batch_size"] * training["sequence_length"]
    if target_tokens % batch_tokens:
        raise ValueError("evaluation targets must align with evaluation batches")
    manifest = load_manifest(data_dir)
    state = loader_state(
        manifest,
        offset,
        training["sequence_length"],
        training["device_batch_size"],
    )
    loader = packed_loader(
        tokenizer,
        training["device_batch_size"],
        training["sequence_length"],
        "val",
        device=device,
        resume_state_dict=state,
        data_dir=data_dir,
    )
    model.eval()
    loss = torch.zeros((), device=device)
    for _ in range(target_tokens // batch_tokens):
        inputs, targets, _ = next(loader)
        loss += model(inputs, targets)
    loss /= target_tokens // batch_tokens
    model.train()
    if not torch.isfinite(loss):
        raise FloatingPointError("model produced a non-finite evaluation loss")
    return loss.item()


def _curve_with(curve, tokens, nll):
    values = [point for point in curve if point["tokens"] != tokens]
    values.append({"tokens": tokens, "nll": nll})
    return sorted(values, key=lambda point: point["tokens"])


def train_candidate(
    study,
    candidate_id,
    target_tokens,
    device_name,
    deadline=None,
    run_name=None,
):
    store, settings, state, configs, candidate, architecture = _context(study, candidate_id)
    if run_name not in {None, "continuation", "independent", "rebuild"}:
        raise ValueError("unknown final training run")
    final_run = run_name in {"continuation", "independent"}
    seed = settings["seed"]
    if run_name == "independent":
        seed += settings["final_seed_offset"]
    device = _runtime(device_name, seed)
    _verify_runtime(state, settings, device)
    tokenizer = get_tokenizer(**configs["tokenizer"])
    if (
        tokenizer.vocab_size != architecture.vocab_size
        or tokenizer.bos_id != architecture.bos_token_id
        or tokenizer.eos_id != architecture.eos_token_id
    ):
        raise ValueError("candidate architecture and tokenizer differ")
    data_dir = configs["data"].get("output_dir") or str(default_data_dir / "packed")
    manifest = load_manifest(data_dir)
    manifest_hash = manifest_fingerprint(manifest)
    training = settings["training"]
    if run_name is None and target_tokens not in settings["rungs"]:
        raise ValueError("search training target is not a rung")
    if final_run and target_tokens != settings["final_tokens"]:
        raise ValueError("final training target must match the final horizon")
    if run_name == "rebuild" and target_tokens != settings["rungs"][-1]:
        raise ValueError("archive rebuild target must match the final rung")
    if target_tokens % training["batch_tokens"]:
        raise ValueError("training target must align with optimizer batches")

    model = _model(architecture, device, seed)
    parameters = tuple(model.parameters())
    optimizer = model.optimizer(
        training["learning_rate"],
        training["weight_decay"],
        training["optimizer"],
    )
    result_path = None
    if run_name is None:
        checkpoint_dir = candidate / "checkpoint"
        result = next(value for value in store.results() if value["candidate_id"] == candidate_id)
    else:
        run_directory = (
            candidate / "rebuild" if run_name == "rebuild" else candidate / "final" / run_name
        )
        checkpoint_dir = run_directory / "checkpoint"
        result_path = run_directory / "result.json"
        result = (
            json.loads(result_path.read_text(encoding="utf-8"))
            if result_path.is_file()
            else {
                "format_version": 1,
                "run": run_name,
                "seed": seed,
                "status": "pending",
                "trained_tokens": 0,
                "nll_curve": [],
                "final_nll": None,
            }
        )
    checkpoint_step = latest(checkpoint_dir)
    load_directory = checkpoint_dir
    if checkpoint_step is None and run_name == "continuation":
        load_directory = candidate / "checkpoint"
        checkpoint_step = latest(load_directory)
        if checkpoint_step is None:
            raise FileNotFoundError("final continuation requires a retained search checkpoint")
    start_step = 0
    data_state = None
    elapsed_training = 0.0
    curve = list(result.get("nll_curve", []))
    metadata = None
    if checkpoint_step is not None:
        model_state, optimizer_state, metadata = load(load_directory, checkpoint_step, device)
        if (
            metadata["architecture_digest"] != architecture.digest
            or metadata["manifest"] != manifest_hash
            or metadata["training"] != training
            or metadata.get("seed", settings["seed"]) != seed
        ):
            raise ValueError("candidate checkpoint does not match the study")
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        start_step = metadata["step"]
        data_state = metadata["data_state"]
        expected_tokens = start_step * training["batch_tokens"]
        if metadata.get("trained_tokens") != expected_tokens:
            raise ValueError("candidate checkpoint step and trained tokens differ")
        validate_loader_progress(data_state, expected_tokens)
        elapsed_training = metadata["training_seconds"]
        curve = metadata["nll_curve"]

    micro_tokens = training["device_batch_size"] * training["sequence_length"]
    accumulation = training["batch_tokens"] // micro_tokens
    schedule_steps = training["schedule_tokens"] // training["batch_tokens"]
    warmup_steps = training["warmup_tokens"] // training["batch_tokens"]
    target_step = target_tokens // training["batch_tokens"]
    if start_step > target_step:
        raise ValueError("candidate checkpoint exceeds the requested rung")
    if metadata is not None:
        trained_tokens = metadata["trained_tokens"]
        complete = start_step == target_step
        if run_name is None:
            result = store.update_result(
                candidate_id,
                status="ready" if complete else "running",
                rung=trained_tokens
                if trained_tokens in settings["rungs"]
                else result.get("rung", 0),
                trained_tokens=trained_tokens,
                nll_curve=curve,
                training_seconds=elapsed_training,
            )
        else:
            result.update(
                status="completed" if complete else "running",
                trained_tokens=trained_tokens,
                nll_curve=curve,
                final_nll=metadata.get("final_nll"),
                training_seconds=elapsed_training,
            )
            atomic_json(result_path, result)
        if complete:
            return {"complete": True, "trained_tokens": trained_tokens}
    train_data = packed_loader(
        tokenizer,
        training["device_batch_size"],
        training["sequence_length"],
        "train",
        device=device,
        resume_state_dict=data_state,
        data_dir=data_dir,
    )
    batch = next(train_data)
    checkpoint_tokens = list(training["checkpoints"])
    if final_run:
        checkpoint_tokens.extend((2 * settings["rungs"][-1], settings["final_tokens"]))
    checkpoints = {
        tokens // training["batch_tokens"]
        for tokens in checkpoint_tokens
        if start_step * training["batch_tokens"] < tokens <= target_tokens
    }
    checkpoints.add(target_step)
    slices = validation_slices(settings)
    model.train()
    for step in range(start_step, target_step):
        synchronize(device)
        started = time.perf_counter()
        scale = lr_scale(
            step,
            schedule_steps,
            warmup_steps,
            training["minimum_learning_rate_scale"],
        )
        _, _, batch = optimization_step(
            model,
            parameters,
            optimizer,
            train_data,
            batch,
            accumulation,
            training["gradient_clip"],
            training["learning_rate"] * scale,
        )
        synchronize(device)
        elapsed_training += time.perf_counter() - started
        completed = step + 1
        if completed not in checkpoints:
            continue
        trained_tokens = completed * training["batch_tokens"]
        nll = evaluate_nll(
            model,
            tokenizer,
            settings,
            data_dir,
            slices["monitor"]["offset"],
            slices["monitor"]["tokens"],
            device,
        )
        curve = _curve_with(curve, trained_tokens, nll)
        final_nll = result.get("final_nll")
        if final_run and trained_tokens == settings["final_tokens"]:
            final_nll = evaluate_nll(
                model,
                tokenizer,
                settings,
                data_dir,
                slices["final"]["offset"],
                slices["final"]["tokens"],
                device,
            )
        metadata = {
            "format_version": 1,
            "step": completed,
            "trained_tokens": trained_tokens,
            "config": architecture.settings(),
            "architecture_digest": architecture.digest,
            "manifest": manifest_hash,
            "data_state": batch[2],
            "training": training,
            "seed": seed,
            "run": run_name,
            "nll_curve": curve,
            "final_nll": final_nll,
            "training_seconds": elapsed_training,
        }
        save(
            checkpoint_dir,
            completed,
            model.state_dict(),
            optimizer.state_dict(),
            metadata,
        )
        prune_checkpoints(checkpoint_dir, {completed})
        complete = completed == target_step
        if run_name is None:
            store.update_result(
                candidate_id,
                status="ready" if complete else "running",
                rung=trained_tokens if complete else result.get("rung", 0),
                trained_tokens=trained_tokens,
                nll_curve=curve,
                training_seconds=elapsed_training,
            )
            store.write_state(store.state())
        else:
            result.update(
                status="completed" if complete else "running",
                trained_tokens=trained_tokens,
                nll_curve=curve,
                final_nll=final_nll,
                training_seconds=elapsed_training,
            )
            atomic_json(result_path, result)
        if (
            run_name in {None, "rebuild"}
            and not complete
            and deadline is not None
            and time.time() >= deadline
        ):
            if run_name is None:
                store.update_result(candidate_id, status="pending")
            return {"complete": False, "trained_tokens": trained_tokens}
    return {"complete": True, "trained_tokens": target_tokens}


def final_profile_candidate(study, candidate_id, device_name):
    store, settings, state, configs, candidate, architecture = _context(study, candidate_id)
    device = _runtime(device_name, settings["profile"]["seed"])
    _verify_runtime(state, settings, device)
    if device.type != "cuda":
        raise ValueError("final gpu profiles require cuda")
    checkpoint_dir = candidate / "final" / "continuation" / "checkpoint"
    step = latest(checkpoint_dir)
    if step is None:
        raise FileNotFoundError("final continuation checkpoint is missing")
    model_state = load_model(checkpoint_dir, step, device)
    model = SpeckForCausalLM(architecture).to(device)
    model.load_state_dict(model_state)
    final = settings["final_profile"]
    gpu_contract = {
        **settings["profile"],
        "warmups": final["warmups"],
        "requests": final["gpu_requests"],
    }
    eager = _measure_profile(model, architecture, gpu_contract, device)

    length = min(32, architecture.max_position_embeddings)
    fixture = torch.randint(
        architecture.vocab_size,
        (1, length),
        device=device,
        generator=torch.Generator(device=device).manual_seed(gpu_contract["seed"]),
    )
    model.eval()
    with torch.inference_mode():
        eager_output = model(fixture)
        compiled = torch.compile(
            model,
            dynamic=False,
            mode=final["compile_mode"],
        )
        synchronize(device)
        started = time.perf_counter()
        compiled_output = compiled(fixture)
        synchronize(device)
        compilation_seconds = time.perf_counter() - started
    equivalent = torch.allclose(
        eager_output,
        compiled_output,
        atol=final["absolute_tolerance"],
        rtol=final["relative_tolerance"],
    )
    if not equivalent:
        raise ValueError("eager and compiled finalist outputs differ")
    compiled_profile = _measure_profile(compiled, architecture, gpu_contract, device)
    profile_path = candidate / "final" / "profile.json"
    result = (
        json.loads(profile_path.read_text(encoding="utf-8"))
        if profile_path.is_file()
        else {"format_version": 1}
    )
    result.update(
        {
            "format_version": 1,
            "eager_gpu": eager,
            "compiled_gpu": compiled_profile,
            "compilation_seconds": compilation_seconds,
            "outputs_equivalent": equivalent,
        }
    )
    atomic_json(profile_path, result)
    return result


def final_cpu_profile_candidate(study, candidate_id):
    store, settings, state, configs, candidate, architecture = _context(study, candidate_id)
    baseline_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    checkpoint_dir = candidate / "final" / "continuation" / "checkpoint"
    step = latest(checkpoint_dir)
    if step is None:
        raise FileNotFoundError("final continuation checkpoint is missing")
    model_state = load_model(checkpoint_dir, step, "cpu")
    model = SpeckForCausalLM(architecture)
    model.load_state_dict(model_state)
    final = settings["final_profile"]
    cpu_contract = {
        **settings["profile"],
        "device": "cpu",
        "compute_dtype": "float32",
        "warmups": final["warmups"],
        "requests": final["cpu_requests"],
    }
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(final["cpu_threads"])
    try:
        cpu = _measure_profile(
            model,
            architecture,
            cpu_contract,
            torch.device("cpu"),
        )
    finally:
        torch.set_num_threads(previous_threads)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    cpu["memory"].update(
        baseline_rss_bytes=baseline_rss,
        peak_rss_bytes=peak_rss,
    )
    cpu["contract"].update(_cpu_contract(settings))
    profile_path = candidate / "final" / "profile.json"
    result = (
        json.loads(profile_path.read_text(encoding="utf-8"))
        if profile_path.is_file()
        else {"format_version": 1}
    )
    result["cpu"] = cpu
    atomic_json(profile_path, result)
    return result


def study_directory(name):
    if not name or Path(name).name != name:
        raise ValueError("study name must be one path component")
    return Path(base_dir()) / "search" / name


@contextmanager
def study_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ".lock"
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("study already has a running coordinator") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run_child(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        error = subprocess.CalledProcessError(
            result.returncode,
            command,
            output=result.stdout,
            stderr=result.stderr,
        )
        raise error
    return json.loads(result.stdout)


def _child_command(action, store, candidate_id, device, *arguments):
    return [
        sys.executable,
        "-m",
        "scripts.search",
        f"_{action}",
        str(store.directory),
        candidate_id,
        *(str(argument) for argument in arguments),
        "--device",
        device,
    ]


def _failure(error):
    text = "\n".join(
        value
        for value in (
            getattr(error, "stderr", None),
            getattr(error, "stdout", None),
            str(error),
        )
        if value
    ).strip()
    lowered = text.lower()
    if "out of memory" in lowered:
        kind = "oom"
    elif "non-finite" in lowered or "not finite" in lowered:
        kind = "non_finite"
    else:
        kind = "runtime"
    return {
        "type": kind,
        "message": text[-8000:],
        "recorded_at": utc_now(),
    }


def _deadline(store):
    state = store.state()
    hours = state["limits"].get("hours", 0)
    if not hours:
        return None
    elapsed = state["elapsed_seconds"]
    if state.get("active_since") is not None:
        elapsed += max(0.0, time.time() - state["active_since"])
    return time.time() + max(0.0, hours * 3600 - elapsed)


def _budget_expired(store):
    state = store.state()
    hours = state["limits"].get("hours", 0)
    elapsed = state["elapsed_seconds"]
    if state.get("active_since") is not None:
        elapsed += max(0.0, time.time() - state["active_since"])
    return bool(hours and elapsed >= hours * 3600)


def _record(store, candidate_id):
    return next(result for result in store.results() if result["candidate_id"] == candidate_id)


def _run_candidate(store, candidate_id, target, device):
    state = store.state()
    state["current_candidate"] = candidate_id
    store.write_state(state)
    current = _record(store, candidate_id)
    store.update_result(candidate_id, status="running", error=None)
    try:
        if current.get("feasibility", {}).get("status") != "passed":
            run_child(_child_command("check", store, candidate_id, device))
        current = _record(store, candidate_id)
        if "latency" not in (current.get("profile") or {}):
            run_child(_child_command("profile", store, candidate_id, device))
        command = _child_command("train", store, candidate_id, device, target)
        deadline = _deadline(store)
        if deadline is not None:
            command.extend(("--deadline", str(deadline)))
        run_child(command)
    except (subprocess.CalledProcessError, RuntimeError, ValueError) as error:
        store.update_result(candidate_id, status="failed", error=_failure(error))
        prune_checkpoints(store.candidate_path(candidate_id) / "checkpoint")
    finally:
        state = store.state()
        state["current_candidate"] = None
        store.write_state(state)
    return _record(store, candidate_id)


def _generation_results(store, generation):
    return [result for result in store.results() if result["generation"] == generation]


def _check_generation_space(store, plans, settings):
    checkpoint_bytes = sum(
        architecture_metrics(plan.architecture, settings)["weight_bytes"] * 4 for plan in plans
    )
    free = shutil.disk_usage(store.directory).free
    if free < checkpoint_bytes:
        raise OSError(
            f"search generation needs {checkpoint_bytes} checkpoint bytes but only {free} are free"
        )


def _plan_generation(store, baseline, settings, generation):
    all_results = store.results()
    current = _generation_results(store, generation)
    if len(current) == settings["generation_size"]:
        state = store.state()
        state["phase"] = "screen"
        store.write_state(state)
        return current
    existing = [result for result in all_results if result["generation"] != generation]
    if generation == 0:
        plans = initial_generation(
            baseline,
            settings,
            existing_digests=(result["digest"] for result in existing),
        )
    else:
        architectures = store.architectures()
        archive = []
        for result in existing:
            if result.get("trained_tokens", 0) >= settings["rungs"][1]:
                archive.append(
                    {
                        **result,
                        "architecture": architectures[result["candidate_id"]].settings(),
                    }
                )
        plans = later_generation(
            normalize_baseline(baseline, settings),
            archive,
            settings,
            generation,
            existing_digests=(result["digest"] for result in existing),
        )
    _check_generation_space(store, plans, settings)
    return materialize_generation(store, plans, generation, settings)


def _score_rung(store, generation, rung, next_horizon):
    cohort = [
        result
        for result in _generation_results(store, generation)
        if result["status"] != "failed" and result.get("rung", 0) >= rung
    ]
    scored = score_candidates(cohort, next_horizon)
    for result in scored:
        scores_by_rung = dict(result.get("scores_by_rung", {}))
        scores_by_rung[str(rung)] = result["scores"]
        store.update_result(
            result["candidate_id"],
            forecast=result.get("forecast"),
            scores=result["scores"],
            scores_by_rung=scores_by_rung,
        )
    return [_record(store, result["candidate_id"]) for result in scored]


def _rescore_archive(store, rung, next_horizon, update_current=False):
    evidence = []
    for result in store.results():
        if result["status"] == "failed" or result.get("trained_tokens", 0) < rung:
            continue
        value = json.loads(json.dumps(result))
        value["nll_curve"] = [point for point in value["nll_curve"] if point["tokens"] <= rung]
        evidence.append(value)
    scored = score_candidates(evidence, next_horizon)
    for result in scored:
        scores_by_rung = dict(result.get("scores_by_rung", {}))
        scores_by_rung[str(rung)] = result["scores"]
        changes = {"scores_by_rung": scores_by_rung}
        if update_current:
            changes.update(
                forecast=result.get("forecast"),
                scores=result["scores"],
            )
        store.update_result(result["candidate_id"], **changes)
    return scored


def _promote(store, cohort, winners):
    winner_ids = {result["candidate_id"] for result in winners}
    for result in cohort:
        candidate_id = result["candidate_id"]
        if candidate_id in winner_ids:
            store.update_result(candidate_id, status="pending")
        else:
            store.update_result(candidate_id, status="completed")
            prune_checkpoints(store.candidate_path(candidate_id) / "checkpoint")


def _checkpoint_ready(directory, step, digest, trained_tokens):
    directory = Path(directory)
    if latest(directory) != step:
        return False
    paths = (
        directory / f"model_{step:06d}.pt",
        directory / f"optimizer_{step:06d}.pt",
        directory / f"metadata_{step:06d}.json",
    )
    if not all(path.is_file() for path in paths):
        return False
    try:
        metadata = json.loads(paths[2].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("trained_tokens") == trained_tokens
        and metadata.get("architecture_digest") == digest
    )


def _install_checkpoint(source, destination, step):
    source = Path(source)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    prune_checkpoints(destination)
    names = (
        f"model_{step:06d}.pt",
        f"optimizer_{step:06d}.pt",
        f"metadata_{step:06d}.json",
    )
    for name in names:
        temporary = destination / f"{name}.tmp"
        shutil.copy2(source / name, temporary)
        os.replace(temporary, destination / name)
    complete = destination / f"complete_{step:06d}"
    temporary = destination / f"complete_{step:06d}.tmp"
    temporary.write_text("complete\n", encoding="utf-8")
    os.replace(temporary, complete)


def _ensure_archive_checkpoint(store, result, settings, device, deadline=None):
    candidate = store.candidate_path(result["candidate_id"])
    checkpoint_dir = candidate / "checkpoint"
    rebuild_root = candidate / "rebuild"
    rebuild_dir = rebuild_root / "checkpoint"
    final_tokens = settings["rungs"][-1]
    final_step = final_tokens // settings["training"]["batch_tokens"]
    if _checkpoint_ready(
        checkpoint_dir,
        final_step,
        result["digest"],
        final_tokens,
    ):
        if rebuild_root.exists():
            shutil.rmtree(rebuild_root)
        return True
    command = _child_command(
        "rebuild",
        store,
        result["candidate_id"],
        device,
        final_tokens,
    )
    if deadline is not None:
        command.extend(("--deadline", str(deadline)))
    rebuilt = run_child(command)
    if not rebuilt["complete"]:
        return False
    if not _checkpoint_ready(
        rebuild_dir,
        final_step,
        result["digest"],
        final_tokens,
    ):
        raise RuntimeError("archive rebuild produced an invalid checkpoint")
    _install_checkpoint(rebuild_dir, checkpoint_dir, final_step)
    if not _checkpoint_ready(
        checkpoint_dir,
        final_step,
        result["digest"],
        final_tokens,
    ):
        raise RuntimeError("archive checkpoint installation failed")
    shutil.rmtree(rebuild_root)
    return True


def _run_phase(store, settings, generation, phase, device):
    index = {"screen": 0, "develop": 1, "confirm": 2}[phase]
    target = settings["rungs"][index]
    candidates = _generation_results(store, generation)
    for result in candidates:
        if result["status"] == "failed" or result.get("rung", 0) >= target:
            continue
        if result["status"] not in {"pending", "running"}:
            continue
        if _budget_expired(store):
            return False
        updated = _run_candidate(store, result["candidate_id"], target, device)
        if updated["status"] == "pending" and updated.get("rung", 0) < target:
            return False
    candidates = _generation_results(store, generation)
    if any(
        result["status"] in {"pending", "running"} and result.get("rung", 0) < target
        for result in candidates
    ):
        return False
    next_horizon = (
        settings["rungs"][index + 1]
        if index + 1 < len(settings["rungs"])
        else settings["final_tokens"]
    )
    scored = _score_rung(store, generation, target, next_horizon)
    if phase != "confirm":
        winners = promotion_for_rung(scored, settings, target)
        _promote(store, scored, winners)
        state = store.state()
        state["phase"] = "develop" if phase == "screen" else "confirm"
        store.write_state(state)
        return True
    for result in scored:
        store.update_result(result["candidate_id"], status="confirmed")
    _rescore_archive(
        store,
        settings["rungs"][1],
        settings["rungs"][2],
    )
    _rescore_archive(
        store,
        settings["rungs"][2],
        settings["final_tokens"],
        update_current=True,
    )
    confirmed = [result for result in store.results() if result["status"] == "confirmed"]
    retained = retained_checkpoint_candidates(confirmed)
    final_step = settings["rungs"][-1] // settings["training"]["batch_tokens"]
    for result in store.results():
        if result["status"] != "confirmed":
            continue
        checkpoint_dir = store.candidate_path(result["candidate_id"]) / "checkpoint"
        if result["candidate_id"] in retained:
            if not _ensure_archive_checkpoint(
                store,
                result,
                settings,
                device,
                _deadline(store),
            ):
                return False
        else:
            rebuild_root = store.candidate_path(result["candidate_id"]) / "rebuild"
            if rebuild_root.exists():
                shutil.rmtree(rebuild_root)
        keep = {final_step} if result["candidate_id"] in retained else set()
        prune_checkpoints(checkpoint_dir, keep)
    state = store.state()
    state["phase"] = "complete"
    state["completed_generations"] = state.get("completed_generations", 0) + 1
    store.write_state(state)
    return True


def run_study(experiment, name, hours, generations, device):
    directory = study_directory(name)
    settings = load_search_settings(Path(experiment) / "search.json")
    configs, inputs = _study_inputs(experiment)
    _verify_inputs(inputs)
    provenance = {
        "inputs": inputs,
        "runtime": _runtime_contract(settings, torch.device(device)),
    }
    baseline = ArchitectureConfig.from_dict(configs["model"])
    with study_lock(directory):
        store = open_study(
            directory,
            experiment,
            settings,
            hours,
            generations,
            provenance,
        )
        state = store.state()
        state["active_since"] = time.time()
        state["status"] = "running"
        state.setdefault("completed_generations", 0)
        store.write_state(state)
        try:
            while True:
                state = store.state()
                if _budget_expired(store):
                    state["status"] = "stopped"
                    store.write_state(state)
                    break
                generation_limit = state["limits"].get("generations", 0)
                if state["phase"] == "complete":
                    if generation_limit and state["completed_generations"] >= generation_limit:
                        state["status"] = "stopped"
                        store.write_state(state)
                        break
                    state["generation"] += 1
                    state["phase"] = "planning"
                    store.write_state(state)
                if state["phase"] == "planning":
                    _plan_generation(store, baseline, settings, state["generation"])
                    continue
                if state["phase"] in {"screen", "develop", "confirm"}:
                    progressed = _run_phase(
                        store,
                        settings,
                        state["generation"],
                        state["phase"],
                        device,
                    )
                    if not progressed:
                        state = store.state()
                        state["status"] = "stopped"
                        store.write_state(state)
                        break
                    continue
                raise RuntimeError(f"unknown search phase: {state['phase']}")
        except BaseException:
            state = store.state()
            state["status"] = "failed"
            state["current_candidate"] = None
            store.write_state(state)
            raise
        finally:
            state = store.state()
            if state.get("active_since") is not None:
                state["elapsed_seconds"] += max(0.0, time.time() - state["active_since"])
            state["active_since"] = None
            store.write_state(state)
    return status_snapshot(store)


def human_status(snapshot):
    lines = [
        f"study {snapshot['status']} | phase {snapshot['phase']} | {snapshot['elapsed_seconds'] / 3600:.2f} hours",
        f"generation {snapshot['generation']} | current {snapshot['current_candidate']['candidate_id'] if snapshot['current_candidate'] else 'none'}",
    ]
    status_counts = ", ".join(
        f"{name} {count}" for name, count in snapshot["counts"]["status"].items()
    )
    rung_counts = ", ".join(f"{name} {count}" for name, count in snapshot["counts"]["rung"].items())
    lines.append(f"status counts | {status_counts or 'none'}")
    lines.append(f"rung counts | {rung_counts or 'none'}")
    for lane in ("quality", "balanced", "efficiency"):
        leader = snapshot["leaders"].get(lane)
        if leader:
            lines.append(f"{lane} leader | {leader['candidate_id']} | score {leader['score']:.4f}")
    current = snapshot["current_candidate"]
    if current:
        if current.get("nll_curve"):
            point = current["nll_curve"][-1]
            lines.append(f"current nll | {point['nll']:.5f} at {point['tokens']} tokens")
        if current.get("forecast"):
            lines.append(
                f"projected nll | {current['forecast']['projected_nll']:.5f} at {current['forecast']['projected_tokens']} tokens"
            )
        profile = current.get("profile") or {}
        if "latency" in profile:
            lines.append(
                f"prefill 2048 p50 | {profile['latency']['prefill_2048']['p50_seconds']:.6f} seconds"
            )
        if profile.get("memory", {}).get("peak_vram_bytes") is not None:
            lines.append(f"peak vram | {profile['memory']['peak_vram_bytes']} bytes")
    lines.append(f"retained checkpoints | {snapshot['checkpoint_bytes']} bytes")
    return "\n".join(lines)


def _final_efficiency(candidates):
    paths = {
        "prefill_512": ("profile", "eager_gpu", "latency", "prefill_512", "p50_seconds"),
        "prefill_2048": ("profile", "eager_gpu", "latency", "prefill_2048", "p50_seconds"),
        "decode_2048": ("profile", "eager_gpu", "latency", "decode_2048", "p50_seconds"),
        "weight_bytes": ("search_profile", "static", "weight_bytes"),
        "state_bytes": ("search_profile", "static", "state_bytes", "2048"),
        "peak_vram": ("profile", "eager_gpu", "memory", "peak_vram_bytes"),
    }

    def value(candidate, path):
        result = candidate
        for key in path:
            result = result[key]
        return result

    ranks = {
        name: percentile_ranks(
            {candidate_id: value(candidate, path) for candidate_id, candidate in candidates.items()}
        )
        for name, path in paths.items()
    }
    scores = {}
    for candidate_id in candidates:
        latency = statistics.mean(
            ranks[name][candidate_id] for name in ("prefill_512", "prefill_2048", "decode_2048")
        )
        memory = statistics.mean(
            ranks[name][candidate_id] for name in ("weight_bytes", "state_bytes", "peak_vram")
        )
        scores[candidate_id] = (latency + memory) / 2
    return scores


def finalize_study(name, device):
    directory = study_directory(name)
    with study_lock(directory):
        store = StudyStore(directory)
        settings = store.settings()
        _verify_inputs(store.state()["provenance"]["inputs"])
        roles_before = select_finalists(store.results())
        candidates = {}
        for candidate_id in sorted(set(roles_before.values())):
            candidate = store.candidate_path(candidate_id)
            search_result = _record(store, candidate_id)
            if not _ensure_archive_checkpoint(
                store,
                search_result,
                settings,
                device,
            ):
                raise RuntimeError("finalist checkpoint rebuild did not complete")
            for run_name in ("continuation", "independent"):
                result_path = candidate / "final" / run_name / "result.json"
                complete = (
                    result_path.is_file()
                    and json.loads(result_path.read_text(encoding="utf-8")).get("status")
                    == "completed"
                )
                if not complete:
                    run_child(
                        _child_command(
                            "final_train",
                            store,
                            candidate_id,
                            device,
                            run_name,
                            settings["final_tokens"],
                        )
                    )
            profile_path = candidate / "final" / "profile.json"
            stored_profile = (
                json.loads(profile_path.read_text(encoding="utf-8"))
                if profile_path.is_file()
                else {}
            )
            if not {"eager_gpu", "compiled_gpu"} <= stored_profile.keys():
                run_child(_child_command("final_profile", store, candidate_id, device))
                stored_profile = json.loads(profile_path.read_text(encoding="utf-8"))
            expected_cpu = _cpu_contract(settings)
            stored_cpu = stored_profile.get("cpu", {}).get("contract", {})
            if any(stored_cpu.get(key) != value for key, value in expected_cpu.items()):
                run_child(_child_command("final_cpu_profile", store, candidate_id, "cpu"))
                stored_profile = json.loads(profile_path.read_text(encoding="utf-8"))
                stored_cpu = stored_profile.get("cpu", {}).get("contract", {})
                if any(stored_cpu.get(key) != value for key, value in expected_cpu.items()):
                    raise RuntimeError("final cpu profile contract differs from the host")
            runs = {
                run_name: json.loads(
                    (candidate / "final" / run_name / "result.json").read_text(encoding="utf-8")
                )
                for run_name in ("continuation", "independent")
            }
            candidates[candidate_id] = {
                "candidate_id": candidate_id,
                "digest": search_result["digest"],
                "search_scores": search_result["scores"],
                "search_profile": search_result["profile"],
                "verification": aggregate_final_runs(runs, settings["final_tokens"]),
                "profile": json.loads(profile_path.read_text(encoding="utf-8")),
            }

        quality_values = {
            candidate_id: candidate["verification"]["mean_final_nll"]
            for candidate_id, candidate in candidates.items()
        }
        quality_ranks = percentile_ranks(quality_values)
        efficiency = _final_efficiency(candidates)
        balanced = {
            candidate_id: (quality_ranks[candidate_id] + efficiency[candidate_id]) / 2
            for candidate_id in candidates
        }
        roles_after = {
            "quality": min(quality_values, key=lambda key: (quality_values[key], key)),
            "balanced": min(balanced, key=lambda key: (balanced[key], key)),
            "efficiency": min(efficiency, key=lambda key: (efficiency[key], key)),
        }
        report = {
            "format_version": 1,
            "generated_at": utc_now(),
            "roles_before": roles_before,
            "roles_after": roles_after,
            "role_changes": {
                lane: {
                    "before": roles_before[lane],
                    "after": roles_after[lane],
                    "changed": roles_before[lane] != roles_after[lane],
                }
                for lane in ("quality", "balanced", "efficiency")
            },
            "candidates": candidates,
        }
        atomic_json(directory / "finalists.json", report)
        state = store.state()
        state["status"] = "finalized"
        state["phase"] = "finalized"
        store.write_state(state)
        return report


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(
        dest="command",
        required=True,
        metavar="{run,status,finalize}",
        help="search operation",
    )
    run = commands.add_parser(
        "run",
        help="start or resume a search study",
        description="Start or resume a Speck architecture search study.",
    )
    run.add_argument("experiment", help="experiment directory")
    run.add_argument("--name", required=True, help="study name")
    run.add_argument(
        "--hours",
        type=float,
        default=None,
        help="optional total runtime limit in hours",
    )
    run.add_argument(
        "--generations",
        type=int,
        default=None,
        help="optional total generation limit",
    )
    run.add_argument(
        "--device",
        default="cuda",
        help="search device (default: %(default)s)",
    )
    status = commands.add_parser(
        "status",
        help="show the current study status",
        description="Show progress and results for a search study.",
    )
    status.add_argument("name", help="study name")
    status.add_argument(
        "--json",
        action="store_true",
        help="print the complete status snapshot as JSON",
    )
    finalize = commands.add_parser(
        "finalize",
        help="train, profile, and rank the study finalists",
        description="Train, profile, and rank finalists from a completed search study.",
    )
    finalize.add_argument("name", help="study name")
    finalize.add_argument(
        "--device",
        default="cuda",
        help="finalization device (default: %(default)s)",
    )
    check = commands.add_parser("_check")
    check.add_argument("study")
    check.add_argument("candidate")
    check.add_argument("--device", required=True)
    profile = commands.add_parser("_profile")
    profile.add_argument("study")
    profile.add_argument("candidate")
    profile.add_argument("--device", required=True)
    train = commands.add_parser("_train")
    train.add_argument("study")
    train.add_argument("candidate")
    train.add_argument("target", type=int)
    train.add_argument("--device", required=True)
    train.add_argument("--deadline", type=float, default=None)
    rebuild = commands.add_parser("_rebuild")
    rebuild.add_argument("study")
    rebuild.add_argument("candidate")
    rebuild.add_argument("target", type=int)
    rebuild.add_argument("--device", required=True)
    rebuild.add_argument("--deadline", type=float, default=None)
    final_train = commands.add_parser("_final_train")
    final_train.add_argument("study")
    final_train.add_argument("candidate")
    final_train.add_argument("run", choices=("continuation", "independent"))
    final_train.add_argument("target", type=int)
    final_train.add_argument("--device", required=True)
    final_profile = commands.add_parser("_final_profile")
    final_profile.add_argument("study")
    final_profile.add_argument("candidate")
    final_profile.add_argument("--device", required=True)
    final_cpu_profile = commands.add_parser("_final_cpu_profile")
    final_cpu_profile.add_argument("study")
    final_cpu_profile.add_argument("candidate")
    final_cpu_profile.add_argument("--device", required=True)
    return parser


def main():
    args = parser().parse_args()
    if args.command == "run":
        result = run_study(
            args.experiment,
            args.name,
            args.hours,
            args.generations,
            args.device,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "status":
        result = status_snapshot(StudyStore(study_directory(args.name)))
        print(json.dumps(result, indent=2, sort_keys=True) if args.json else human_status(result))
        return
    if args.command == "finalize":
        result = finalize_study(args.name, args.device)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "_check":
        result = check_candidate(args.study, args.candidate, args.device)
    elif args.command == "_profile":
        result = profile_candidate(args.study, args.candidate, args.device)
    elif args.command == "_train":
        result = train_candidate(
            args.study,
            args.candidate,
            args.target,
            args.device,
            args.deadline,
        )
    elif args.command == "_final_train":
        result = train_candidate(
            args.study,
            args.candidate,
            args.target,
            args.device,
            run_name=args.run,
        )
    elif args.command == "_rebuild":
        result = train_candidate(
            args.study,
            args.candidate,
            args.target,
            args.device,
            deadline=args.deadline,
            run_name="rebuild",
        )
    elif args.command == "_final_profile":
        result = final_profile_candidate(args.study, args.candidate, args.device)
    else:
        result = final_cpu_profile_candidate(args.study, args.candidate)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
