"""run resumable architecture searches."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from speck.architecture import ArchitectureConfig
from speck.checkpoint import latest, load, save
from speck.config import load_experiment
from speck.dataloader import manifest_fingerprint, packed_loader
from speck.dataset import default_data_dir, load_manifest
from speck.model import SpeckForCausalLM
from speck.search import (
    StudyStore,
    loader_state,
    prune_checkpoints,
    validation_slices,
)
from speck.tokenizer import get_tokenizer
from speck.train import lr_scale, optimization_step


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
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.set_float32_matmul_precision("high")
    return device


def _context(study, candidate_id):
    store = StudyStore(study)
    settings = store.settings()
    state = store.state()
    configs = load_experiment(state["experiment"], "data", "tokenizer")
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
    store, settings, state, configs, candidate, architecture = _context(
        study, candidate_id
    )
    device = _runtime(device_name, settings["seed"])
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
            [
                model(fixture[:, index : index + 1], state=sequence_state)
                for index in range(length)
            ],
            dim=1,
        )
    tolerance = settings["final_profile"]
    if not torch.allclose(
        full,
        cached,
        atol=tolerance["absolute_tolerance"],
        rtol=tolerance["relative_tolerance"],
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


def profile_candidate(study, candidate_id, device_name):
    store, settings, state, configs, candidate, architecture = _context(
        study, candidate_id
    )
    profile = settings["profile"]
    device = _runtime(device_name, profile["seed"])
    model = _model(architecture, device, profile["seed"])
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
    result = store.results()
    current = next(value for value in result if value["candidate_id"] == candidate_id)
    candidate_profile = dict(current.get("profile") or {})
    candidate_profile.update(
        contract={
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "cpu",
            "parameter_dtype": profile["parameter_dtype"],
            "compute_dtype": profile["compute_dtype"],
            "warmups": profile["warmups"],
            "requests": profile["requests"],
            "seed": profile["seed"],
        },
        latency={name: _distribution(values) for name, values in samples.items()},
        memory={
            "resident_vram_bytes": resident,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
        },
    )
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
        manifest_fingerprint(manifest),
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


def train_candidate(study, candidate_id, target_tokens, device_name, deadline=None):
    store, settings, state, configs, candidate, architecture = _context(
        study, candidate_id
    )
    device = _runtime(device_name, settings["seed"])
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
    if target_tokens not in settings["rungs"]:
        raise ValueError("search training target is not a rung")
    if target_tokens % training["batch_tokens"]:
        raise ValueError("training target must align with optimizer batches")

    model = _model(architecture, device, settings["seed"])
    parameters = tuple(model.parameters())
    optimizer = model.optimizer(
        training["learning_rate"],
        training["weight_decay"],
        training["optimizer"],
    )
    checkpoint_dir = candidate / "checkpoint"
    checkpoint_step = latest(checkpoint_dir)
    start_step = 0
    data_state = None
    elapsed_training = 0.0
    result = next(
        value for value in store.results() if value["candidate_id"] == candidate_id
    )
    curve = list(result.get("nll_curve", []))
    if checkpoint_step is not None:
        model_state, optimizer_state, metadata = load(
            checkpoint_dir, checkpoint_step, device
        )
        if (
            metadata["architecture_digest"] != architecture.digest
            or metadata["manifest"] != manifest_hash
            or metadata["training"] != training
        ):
            raise ValueError("candidate checkpoint does not match the study")
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        start_step = metadata["step"]
        data_state = metadata["data_state"]
        elapsed_training = metadata["training_seconds"]
        curve = metadata["nll_curve"]

    micro_tokens = training["device_batch_size"] * training["sequence_length"]
    accumulation = training["batch_tokens"] // micro_tokens
    schedule_steps = training["schedule_tokens"] // training["batch_tokens"]
    warmup_steps = training["warmup_tokens"] // training["batch_tokens"]
    target_step = target_tokens // training["batch_tokens"]
    if start_step > target_step:
        raise ValueError("candidate checkpoint exceeds the requested rung")
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
    checkpoints = {
        tokens // training["batch_tokens"]
        for tokens in training["checkpoints"]
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
        metadata = {
            "format_version": 1,
            "step": completed,
            "trained_tokens": trained_tokens,
            "config": architecture.settings(),
            "architecture_digest": architecture.digest,
            "manifest": manifest_hash,
            "data_state": batch[2],
            "training": training,
            "nll_curve": curve,
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
        store.update_result(
            candidate_id,
            status="ready" if complete else "running",
            rung=trained_tokens if complete else result.get("rung", 0),
            trained_tokens=trained_tokens,
            nll_curve=curve,
            training_seconds=elapsed_training,
        )
        if not complete and deadline is not None and time.time() >= deadline:
            store.update_result(candidate_id, status="pending")
            return {"complete": False, "trained_tokens": trained_tokens}
    return {"complete": True, "trained_tokens": target_tokens}


def private_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
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
    return parser


def main():
    args = private_parser().parse_args()
    if args.command == "_check":
        result = check_candidate(args.study, args.candidate, args.device)
    elif args.command == "_profile":
        result = profile_candidate(args.study, args.candidate, args.device)
    else:
        result = train_candidate(
            args.study,
            args.candidate,
            args.target,
            args.device,
            args.deadline,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
