"""Benchmark normalized prefill and cached decode speed across small language models."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from speck.architecture import ArchitectureConfig
from speck.checkpoint import load_model
from speck.common import base_dir
from speck.config import load_experiment
from speck.model import SpeckForCausalLM

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

MODELS = {
    "speck": {},
    "supra": {
        "name": "SupraLabs/Supra2-100M-Base",
        "revision": "a664acf32f2210ce1a1d5a85b1a381977f6d5d4b",
        "native_last_logit": True,
    },
    "gptx": {
        "name": "AxiomicLabs/GPT-X2.5-135M",
        "revision": "4b49a8d6986f11989df15de37b44be7d04e634e7",
        "native_last_logit": False,
    },
    "banana": {
        "name": "BananaMind/BananaMind-2-Pro",
        "revision": "c215af4f2cc46d01cba6c5be3c132f4c0e9a9871",
        "native_last_logit": False,
    },
    "smol": {
        "name": "HuggingFaceTB/SmolLM2-135M",
        "revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
        "native_last_logit": True,
    },
}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument(
        "--batch-sizes",
        type=_batch_sizes,
        default=None,
        help="comma-separated batch sizes; defaults to 1 on CPU and 1,32 on CUDA",
    )
    parser.add_argument("--prefill-length", type=int, default=512)
    parser.add_argument("--decode-prefix-length", type=int, default=448)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="compile model forward with max-autotune-no-cudagraphs",
    )
    parser.add_argument("--speck-experiment", default="experiments/Speck1-140M")
    parser.add_argument("--speck-checkpoint-step", type=int, default=76294)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _batch_sizes(value):
    try:
        batches = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("batch sizes must be comma-separated integers") from error
    if not batches or any(batch < 1 for batch in batches):
        raise argparse.ArgumentTypeError("batch sizes must be positive")
    return batches


def _percentile(values, fraction):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() or None


def _speck_identity(experiment, run, checkpoint_dir, checkpoint_step, metadata):
    if metadata.get("step") != checkpoint_step:
        raise ValueError("checkpoint metadata step does not match the requested step")
    source_run = metadata.get("resolved", {}).get("run")
    if not isinstance(source_run, str) or not source_run:
        raise ValueError("checkpoint metadata does not identify its source run")
    model_path = checkpoint_dir / f"model_{checkpoint_step:06d}.pt"
    metadata_path = checkpoint_dir / f"metadata_{checkpoint_step:06d}.json"
    return run, {
        "directory": str(checkpoint_dir.resolve()),
        "step": checkpoint_step,
        "experiment": str(Path(experiment).resolve()),
        "source_run": source_run,
        "source_experiment": metadata["resolved"].get("experiment"),
        "model_sha256": _file_sha256(model_path),
        "metadata_sha256": _file_sha256(metadata_path),
    }


def _cpu_name():
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _cpu_affinity():
    return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None


class ModelRunner:
    def __init__(self, key, device, dtype, experiment, checkpoint_step, compile_model=False):
        self.key = key
        self.device = device
        self.dtype = dtype
        self.spec = MODELS[key]
        self.name = self.spec.get("name")
        self.revision = _git_revision() if key == "speck" else self.spec["revision"]
        self.last_logit_method = "native"
        self.checkpoint = None
        if key == "speck":
            self.model = self._load_speck(experiment, checkpoint_step)
        else:
            from transformers import AutoModelForCausalLM

            kwargs = {
                "revision": self.spec["revision"],
                "trust_remote_code": True,
                "local_files_only": True,
                "dtype": dtype,
            }
            if key in {"supra", "smol"}:
                kwargs["attn_implementation"] = "sdpa"
            self.model = AutoModelForCausalLM.from_pretrained(self.spec["name"], **kwargs)
            if key == "banana":
                self.model.tie_weights()
            self.model.to(device).eval()
            if not self.spec["native_last_logit"]:
                self.last_logit_method = "lm_head_hook"
                self._validate_and_install_last_logit_hook()
        self.parameters = _parameter_count(self.model)
        self._validate_normalized_logits()
        self.forward_model = (
            torch.compile(
                self.model,
                dynamic=False,
                mode="max-autotune-no-cudagraphs",
            )
            if compile_model
            else self.model
        )

    def _load_speck(self, experiment, checkpoint_step):
        configs = load_experiment(experiment, "train")
        checkpoint_dir = Path(
            configs["train"].get("output_dir")
            or Path(base_dir()) / "checkpoints" / configs["train"]["run"]
        ).expanduser()
        metadata_path = checkpoint_dir / f"metadata_{checkpoint_step:06d}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        state = load_model(checkpoint_dir, checkpoint_step, "cpu")
        model = SpeckForCausalLM(ArchitectureConfig.from_dict(metadata["config"]))
        model.load_state_dict(state)
        model.to(self.device, dtype=self.dtype).eval()
        self.name, self.checkpoint = _speck_identity(
            experiment,
            configs["train"]["run"],
            checkpoint_dir,
            checkpoint_step,
            metadata,
        )
        return model

    def _validate_and_install_last_logit_hook(self):
        probe = torch.arange(4, 12, device=self.device)[None]
        with torch.inference_mode():
            expected = self.model(probe, use_cache=False).logits[:, -1:]

        handle = self.model.lm_head.register_forward_pre_hook(
            lambda module, inputs: (inputs[0][:, -1:, :],)
        )
        with torch.inference_mode():
            actual = self.model(probe, use_cache=False).logits
        if not torch.allclose(actual.float(), expected.float(), atol=1e-2, rtol=1e-2):
            handle.remove()
            raise ValueError(f"last-logit hook changed {self.key} output")

    def _validate_normalized_logits(self):
        probe = torch.arange(4, 12, device=self.device)[None]
        with torch.inference_mode():
            if self.key == "speck":
                expected = self.model(probe)[:, -1:]
                actual = self.model(probe, last_token_only=True)
            else:
                actual = self._hf_forward(probe, use_cache=False).logits
                if self.last_logit_method == "native":
                    expected = self.model(probe, use_cache=False).logits[:, -1:]
                else:
                    expected = actual
        if actual.shape[1] != 1 or not torch.allclose(
            actual.float(), expected.float(), atol=1e-2, rtol=1e-2
        ):
            raise ValueError(f"normalized last-logit output failed for {self.key}")

    def _hf_forward(self, tokens, **kwargs):
        if self.spec["native_last_logit"]:
            kwargs["logits_to_keep"] = 1
        return self.model(input_ids=tokens, **kwargs)

    def prefill(self, tokens, state_length=None):
        if self.key == "speck":
            state = self.model.state(
                batch_size=tokens.size(0),
                length=state_length or tokens.size(1),
                device=self.device,
                dtype=self.dtype,
            )
            logits = self.forward_model(tokens, state=state, last_token_only=True)
            return logits, state
        output = self._hf_forward(tokens, use_cache=True)
        return output.logits, output.past_key_values

    def decode(self, tokens, state):
        if self.key == "speck":
            return self.forward_model(tokens, state=state, last_token_only=True), state
        output = self._hf_forward(tokens, past_key_values=state, use_cache=True)
        return output.logits, output.past_key_values


def _duration(call, device):
    _synchronize(device)
    started = time.perf_counter()
    value = call()
    _synchronize(device)
    return value, time.perf_counter() - started


def _measure_prefill(runner, tokens, warmups, repeats):
    for _ in range(warmups):
        runner.prefill(tokens)
    _synchronize(runner.device)
    if runner.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runner.device)
    durations = [
        _duration(lambda: runner.prefill(tokens), runner.device)[1] for _ in range(repeats)
    ]
    median = statistics.median(durations)
    token_count = tokens.numel()
    return {
        "tokens_per_second": token_count / median,
        "milliseconds_per_batch": median * 1000,
        "duration_seconds": durations,
        "duration_iqr_seconds": [
            _percentile(durations, 0.25),
            _percentile(durations, 0.75),
        ],
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(runner.device) if runner.device.type == "cuda" else None
        ),
    }


def _decode_stream(runner, prefix, decode_tokens):
    logits, state = runner.prefill(prefix, prefix.size(1) + decode_tokens)
    token = logits[:, -1].argmax(dim=-1, keepdim=True)
    for _ in range(decode_tokens):
        logits, state = runner.decode(token, state)
        token = logits[:, -1].argmax(dim=-1, keepdim=True)


def _measure_decode(runner, prefix, decode_tokens, warmups, repeats):
    for _ in range(warmups):
        _decode_stream(runner, prefix, decode_tokens)
    _synchronize(runner.device)
    if runner.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runner.device)
    # Prefix filling is repeated to create fresh caches but excluded from the decode metric.
    prefix_durations = []
    decode_durations = []
    for _ in range(repeats):
        (logits, state), prefix_duration = _duration(
            lambda: runner.prefill(prefix, prefix.size(1) + decode_tokens), runner.device
        )
        token = logits[:, -1].argmax(dim=-1, keepdim=True)

        def decode_only():
            nonlocal token, state
            for _ in range(decode_tokens):
                logits, state = runner.decode(token, state)
                token = logits[:, -1].argmax(dim=-1, keepdim=True)

        _, decode_duration = _duration(decode_only, runner.device)
        prefix_durations.append(prefix_duration)
        decode_durations.append(decode_duration)
        del decode_only, logits, state, token
    median = statistics.median(decode_durations)
    generated = prefix.size(0) * decode_tokens
    return {
        "tokens_per_second": generated / median,
        "milliseconds_per_step": median * 1000 / decode_tokens,
        "duration_seconds": decode_durations,
        "duration_iqr_seconds": [
            _percentile(decode_durations, 0.25),
            _percentile(decode_durations, 0.75),
        ],
        "untimed_prefix_seconds": prefix_durations,
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(runner.device) if runner.device.type == "cuda" else None
        ),
    }


def run(args):
    if args.threads < 1 or args.prefill_length < 1 or args.decode_prefix_length < 1:
        raise ValueError("threads and sequence lengths must be positive")
    if args.decode_tokens < 1:
        raise ValueError("decode tokens must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    batches = args.batch_sizes or ((1,) if device.type == "cpu" else (1, 32))
    warmups = args.warmups if args.warmups is not None else (2 if device.type == "cpu" else 50)
    repeats = args.repeats if args.repeats is not None else (7 if device.type == "cpu" else 20)
    if warmups < 0 or repeats < 1:
        raise ValueError("warmups must be non-negative and repeats must be positive")

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("high")
    runner = ModelRunner(
        args.model,
        device,
        dtype,
        args.speck_experiment,
        args.speck_checkpoint_step,
        args.compile,
    )

    measurements = []
    decode_warmups = max(5, warmups // 2)
    decode_repeats = repeats
    with torch.inference_mode():
        for batch_size in batches:
            generator = torch.Generator(device=device).manual_seed(args.seed + batch_size)
            prefill_tokens = torch.randint(
                4,
                32_000,
                (batch_size, args.prefill_length),
                generator=generator,
                device=device,
            )
            prefix_tokens = torch.randint(
                4,
                32_000,
                (batch_size, args.decode_prefix_length),
                generator=generator,
                device=device,
            )
            measurements.append(
                {
                    "batch_size": batch_size,
                    "prefill": _measure_prefill(runner, prefill_tokens, warmups, repeats),
                    "decode": _measure_decode(
                        runner,
                        prefix_tokens,
                        args.decode_tokens,
                        decode_warmups,
                        decode_repeats,
                    ),
                }
            )

    return {
        "protocol": {
            "name": "normalized causal LM inference",
            "prefill_length": args.prefill_length,
            "decode_prefix_length": args.decode_prefix_length,
            "decode_tokens": args.decode_tokens,
            "prefill_warmups": warmups,
            "prefill_repeats": repeats,
            "decode_warmups": decode_warmups,
            "decode_repeats": decode_repeats,
            "seed": args.seed,
            "logits": "last token only",
            "cache": "Speck preallocated state; Hugging Face model-native dynamic cache",
            "compiled": args.compile,
            "tokenization": False,
            "token_selection": "argmax included in decode timing",
        },
        "model": {
            "key": args.model,
            "name": runner.name,
            "revision": runner.revision,
            "parameters": runner.parameters,
            "last_logit_method": runner.last_logit_method,
            "checkpoint": runner.checkpoint,
        },
        "environment": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else _cpu_name(),
            "dtype": str(dtype).removeprefix("torch."),
            "threads": args.threads,
            "cpu_count": os.cpu_count(),
            "cpu_affinity": _cpu_affinity(),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "cuda": torch.version.cuda,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        },
        "measurements": measurements,
    }


def main():
    args = arguments()
    report = run(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
