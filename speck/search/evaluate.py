"""fixed-fidelity quality, inference, and memory evaluation."""

import math
import statistics
import time
from dataclasses import dataclass
from typing import Any

import torch

from speck.dataloader import packed_loader
from speck.dataset import load_manifest
from speck.model import Config, SpeckForCausalLM
from speck.search.architecture import kv_bytes_per_token, parameter_count
from speck.train import lr_scale, optimization_step


@dataclass(frozen=True)
class QualitySettings:
    data_dir: str
    train_tokens: int
    batch_tokens: int
    device_batch_size: int
    sequence_length: int
    eval_every_tokens: int
    eval_batch_size: int
    eval_tokens: int
    lr: float
    min_lr: float
    warmup_steps: int
    weight_decay: float
    grad_clip: float
    optimizer: str
    compile: bool = True
    batch_curriculum: bool = False

    def __post_init__(self):
        positive = (
            self.train_tokens,
            self.batch_tokens,
            self.device_batch_size,
            self.sequence_length,
            self.eval_every_tokens,
            self.eval_batch_size,
            self.eval_tokens,
        )
        if any(value < 1 for value in positive):
            raise ValueError("quality token and batch settings must be positive")
        if self.train_tokens % self.batch_tokens:
            raise ValueError("train tokens must be divisible by batch tokens")
        micro_tokens = self.device_batch_size * self.sequence_length
        if self.batch_tokens % micro_tokens:
            raise ValueError("batch tokens must be divisible by micro batch tokens")
        if self.batch_curriculum:
            curriculum_batches = (self.batch_tokens // 4, self.batch_tokens // 2)
            if any(
                batch_tokens < micro_tokens or batch_tokens % micro_tokens
                for batch_tokens in curriculum_batches
            ):
                raise ValueError("curriculum batches must be divisible by micro batch tokens")
        if self.warmup_steps < 0:
            raise ValueError("warmup steps cannot be negative")

    @classmethod
    def from_dict(cls, settings):
        return cls(**settings)


@dataclass(frozen=True)
class InferenceSettings:
    contexts: tuple[int, ...] = (512, 2048)
    warmup_samples: int = 5
    samples: int = 20
    cache_dtype_bytes: int = 2

    def __post_init__(self):
        contexts = tuple(sorted(set(self.contexts)))
        object.__setattr__(self, "contexts", contexts)
        if not contexts or contexts[0] < 1:
            raise ValueError("inference contexts must be positive")
        if self.warmup_samples < 0 or self.samples < 1:
            raise ValueError("invalid inference sample count")
        if self.cache_dtype_bytes < 1:
            raise ValueError("cache dtype bytes must be positive")

    @classmethod
    def from_dict(cls, settings):
        values = dict(settings)
        values["contexts"] = tuple(values.get("contexts", (512, 2048)))
        return cls(**values)


@dataclass(frozen=True)
class QuantizationSettings:
    bits: int = 4
    group_size: int = 128
    scale_bytes: int = 2
    unquantized_bytes: int = 2

    def __post_init__(self):
        if not 1 <= self.bits <= 8:
            raise ValueError("quantization bits must be between one and eight")
        if self.group_size < 1 or self.scale_bytes < 1 or self.unquantized_bytes < 1:
            raise ValueError("quantization layout values must be positive")

    @classmethod
    def from_dict(cls, settings):
        return cls(**settings)


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(values, fraction):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _latency(samples):
    return {
        "samples_ms": samples,
        "mean_ms": statistics.mean(samples),
        "p50_ms": statistics.median(samples),
        "p90_ms": _percentile(samples, 0.9),
    }


def _measure(call, device):
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        return start.elapsed_time(end)
    started = time.perf_counter()
    call()
    return (time.perf_counter() - started) * 1000


@torch.no_grad()
def _validation_loss(
    model,
    tokenizer,
    data_dir,
    manifest,
    batch_size,
    sequence_length,
    token_limit,
    device,
):
    loader = packed_loader(
        tokenizer,
        batch_size,
        sequence_length,
        "val",
        device=device,
        data_dir=data_dir,
    )
    tokens = min(manifest["splits"]["val"]["tokens"], token_limit)
    steps = max(1, tokens // (batch_size * sequence_length))
    loss = torch.zeros((), device=device)
    model.eval()
    for _ in range(steps):
        inputs, targets, _ = next(loader)
        loss += model(inputs, targets)
    model.train()
    return (loss / steps).item(), steps * batch_size * sequence_length


def evaluate_quality(config, tokenizer, settings, device, seed):
    if settings.sequence_length > config.max_position_embeddings:
        raise ValueError("quality sequence exceeds model context")
    manifest = load_manifest(settings.data_dir)
    if manifest["tokenizer"]["fingerprint"] != tokenizer.fingerprint():
        raise ValueError("quality dataset and tokenizer do not match")
    micro_tokens = settings.device_batch_size * settings.sequence_length
    if manifest["splits"]["train"]["tokens"] <= settings.train_tokens + micro_tokens:
        raise ValueError("packed dataset is too small for the quality budget")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    model = SpeckForCausalLM(config).to(device)
    model.init_weights()
    parameters = tuple(model.parameters())
    optimizer = model.optimizer(
        settings.lr, settings.weight_decay, settings.optimizer
    )
    train_model: Any = model
    if settings.compile:
        train_model = torch.compile(
            model, dynamic=False, mode="max-autotune-no-cudagraphs"
        )
    loader = packed_loader(
        tokenizer,
        settings.device_batch_size,
        settings.sequence_length,
        "train",
        device=device,
        data_dir=settings.data_dir,
    )
    batch = next(loader)
    nominal_steps = settings.train_tokens // settings.batch_tokens
    quarter_tokens = math.ceil(nominal_steps / 4) * settings.batch_tokens
    train_curve = []
    validation_curve = []

    def validate(step, tokens):
        loss, evaluated_tokens = _validation_loss(
            train_model,
            tokenizer,
            settings.data_dir,
            manifest,
            settings.eval_batch_size,
            settings.sequence_length,
            settings.eval_tokens,
            device,
        )
        validation_curve.append({"step": step, "tokens": tokens, "loss": loss})
        return evaluated_tokens

    evaluated_tokens = validate(0, 0)
    next_validation = settings.eval_every_tokens
    durations = []
    trained_tokens = 0
    step = 0
    while trained_tokens < settings.train_tokens:
        if settings.batch_curriculum and trained_tokens < quarter_tokens:
            batch_tokens = settings.batch_tokens // 4
        elif settings.batch_curriculum and trained_tokens < 2 * quarter_tokens:
            batch_tokens = settings.batch_tokens // 2
        else:
            batch_tokens = settings.batch_tokens
        if trained_tokens + batch_tokens > settings.train_tokens:
            raise ValueError("quality schedule does not end at the fixed token budget")
        accumulation = batch_tokens // micro_tokens
        schedule_step = trained_tokens / settings.batch_tokens
        scale = lr_scale(
            schedule_step, nominal_steps, settings.warmup_steps, settings.min_lr
        )
        _synchronize(device)
        started = time.perf_counter()
        loss, grad_norm, batch = optimization_step(
            train_model,
            parameters,
            optimizer,
            loader,
            batch,
            accumulation,
            settings.grad_clip,
            settings.lr * scale,
        )
        _synchronize(device)
        durations.append(time.perf_counter() - started)
        step += 1
        trained_tokens += batch_tokens
        train_curve.append(
            {
                "step": step,
                "tokens": trained_tokens,
                "loss": loss.item(),
                "grad_norm": float(grad_norm),
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        if trained_tokens >= next_validation or trained_tokens == settings.train_tokens:
            evaluated_tokens = validate(step, trained_tokens)
            while next_validation <= trained_tokens:
                next_validation += settings.eval_every_tokens

    return {
        "validation_nll": validation_curve[-1]["loss"],
        "train_curve": train_curve,
        "validation_curve": validation_curve,
        "geometry": {
            "train_tokens": settings.train_tokens,
            "batch_tokens": settings.batch_tokens,
            "device_batch_size": settings.device_batch_size,
            "sequence_length": settings.sequence_length,
            "final_accumulation": settings.batch_tokens // micro_tokens,
            "eval_tokens": evaluated_tokens,
            "batch_curriculum": settings.batch_curriculum,
        },
        "performance": {
            "training_seconds": sum(durations),
            "tokens_per_second": settings.train_tokens / sum(durations),
            "step_seconds": durations,
        },
    }


@torch.inference_mode()
def evaluate_inference(config, settings, device, seed):
    if settings.contexts[-1] + 1 > config.max_position_embeddings:
        raise ValueError("decode context exceeds model context")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    model = SpeckForCausalLM(config).to(device)
    model.init_weights()
    model.eval()
    results = {}
    model_allocated = (
        torch.cuda.memory_allocated(device) if device.type == "cuda" else None
    )

    for context in settings.contexts:
        tokens = torch.randint(config.vocab_size, (1, context), device=device)
        next_token = torch.randint(config.vocab_size, (1, 1), device=device)
        cache = model.cache(length=context + 1)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for _ in range(settings.warmup_samples):
            cache.position = 0
            model(tokens, cache=cache)
        prefill_samples = []
        for _ in range(settings.samples):
            cache.position = 0
            prefill_samples.append(
                _measure(lambda: model(tokens, cache=cache), device)
            )

        cache.position = 0
        model(tokens, cache=cache)
        for _ in range(settings.warmup_samples):
            cache.position = context
            model(next_token, cache=cache)
        decode_samples = []
        for _ in range(settings.samples):
            cache.position = context
            decode_samples.append(
                _measure(lambda: model(next_token, cache=cache), device)
            )

        peak = (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        )
        results[str(context)] = {
            "prefill": _latency(prefill_samples),
            "decode": _latency(decode_samples),
            "peak_allocated_bytes": peak,
            "cache_allocated_bytes": sum(
                tensor.numel() * tensor.element_size()
                for tensor in cache.keys + cache.values
            ),
        }
        del cache
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "contexts": results,
        "batch_size": 1,
        "warmup_samples": settings.warmup_samples,
        "samples": settings.samples,
        "model_allocated_bytes": model_allocated,
        "kv_cache_bytes_per_token": kv_bytes_per_token(
            config, settings.cache_dtype_bytes
        ),
    }


def quantized_weight_bytes(config, settings):
    with torch.device("meta"):
        model = SpeckForCausalLM(config)
    total = 0
    breakdown = []
    for name, parameter in model.named_parameters():
        if parameter.ndim == 2:
            rows, columns = parameter.shape
            packed = rows * math.ceil(columns * settings.bits / 8)
            groups = rows * math.ceil(columns / settings.group_size)
            scales = groups * settings.scale_bytes
            size = packed + scales
            kind = "quantized"
        else:
            packed = 0
            scales = 0
            size = parameter.numel() * settings.unquantized_bytes
            kind = "unquantized"
        total += size
        breakdown.append(
            {
                "name": name,
                "kind": kind,
                "elements": parameter.numel(),
                "packed_bytes": packed,
                "scale_bytes": scales,
                "total_bytes": size,
            }
        )
    return {
        "total_bytes": total,
        "bits": settings.bits,
        "group_size": settings.group_size,
        "scale_bytes": settings.scale_bytes,
        "unquantized_bytes": settings.unquantized_bytes,
        "parameters": parameter_count(config),
        "breakdown": breakdown,
    }


def objective_values(quality, inference, quantization):
    objectives = {
        "quality.validation_nll": quality["validation_nll"],
        "memory.kv_cache_bytes_per_token": inference["kv_cache_bytes_per_token"],
        "memory.quantized_weight_bytes": quantization["total_bytes"],
    }
    for context, metrics in inference["contexts"].items():
        objectives[f"prefill.ms.context_{context}"] = metrics["prefill"]["p50_ms"]
        objectives[f"decode.ms_per_token.context_{context}"] = metrics["decode"]["p50_ms"]
        objectives[f"memory.inference_peak_bytes.context_{context}"] = metrics[
            "peak_allocated_bytes"
        ]
    return objectives
