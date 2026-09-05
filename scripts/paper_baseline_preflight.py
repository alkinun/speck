"""Qualify both Paper 1 baseline arms on the frozen RTX 3090 training/export path."""

import argparse
import gc
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import save_file

from scripts.model_publish import (
    prepare_current_release_code,
    release_config,
    release_state,
    validate_export,
    validate_parity,
)
from speck.config import load_experiment
from speck.model import build_model
from speck.paper_baseline import file_sha256, load_matrix
from speck.tokenizer import get_tokenizer


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def command_output(command, cwd=None):
    result = subprocess.run(command, cwd=cwd, capture_output=True, check=True, text=True)
    return result.stdout.strip()


def repository_revision():
    root = Path(__file__).resolve().parents[1]
    if command_output(["git", "status", "--porcelain"], cwd=root):
        raise ValueError("baseline preflight requires a clean repository")
    return command_output(["git", "rev-parse", "HEAD"], cwd=root)


def atomic_json(path, value):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _maximum_error(actual, expected):
    return (actual.float() - expected.float()).abs().max().item()


def _file_inventory(directory):
    files = {}
    for path in sorted(candidate for candidate in directory.iterdir() if candidate.is_file()):
        files[path.name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    return files


def _temperature_c():
    return int(
        command_output(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ]
        ).splitlines()[0]
    )


def _wait_for_temperature(maximum_c=50, timeout_seconds=600, poll_seconds=5):
    started = time.monotonic()
    temperature = _temperature_c()
    while temperature > maximum_c:
        waited = time.monotonic() - started
        if waited >= timeout_seconds:
            raise RuntimeError(
                f"GPU did not cool to the frozen {maximum_c}C start limit; "
                f"last temperature was {temperature}C"
            )
        print(f"waiting for GPU to cool: {temperature}C > {maximum_c}C", flush=True)
        time.sleep(poll_seconds)
        temperature = _temperature_c()
    return temperature, time.monotonic() - started


def _export_case(state, metadata, directory):
    prepare_current_release_code(directory)
    save_file(release_state(state), directory / "model.safetensors", metadata={"format": "pt"})
    (directory / "config.json").write_text(
        json.dumps(release_config(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (directory / "speck_source.json").write_text(
        json.dumps(
            {
                "format": "speck_export_source",
                "format_version": 1,
                "type": "random_weight_hardware_preflight",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        validate_export(directory, metadata)
        parity = validate_parity(directory, state, metadata)
        return {"passed": True, "parity": parity, "files": _file_inventory(directory)}
    except Exception as error:
        return {
            "passed": False,
            "failure_type": type(error).__name__,
            "failure": str(error),
            "files": _file_inventory(directory),
        }


def preflight_arm(arm, output_root, device):
    experiment = output_root / "arms" / arm["id"]
    configs = load_experiment(experiment, "model", "tokenizer", "train")
    train = configs["train"]
    tokenizer = get_tokenizer(**configs["tokenizer"])
    start_temperature, thermal_wait_seconds = _wait_for_temperature()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = build_model(
        configs["model"],
        tokenizer.vocab_size,
        tokenizer.bos_id,
        tokenizer.eos_id,
        loss_backend=train["loss_backend"],
    ).to(device)
    model.init_weights()
    if model.parameter_count() != arm["parameters"]:
        raise ValueError("preflight model parameters do not match the baseline matrix")
    optimizer = model.optimizer(train["lr"], train["weight_decay"], train["optimizer"])
    compiled = torch.compile(
        model,
        dynamic=False,
        options={
            "max_autotune": True,
            "coordinate_descent_tuning": True,
            "aggressive_fusion": True,
        },
    )
    compile_step = getattr(optimizer, "compile_step", None)
    if compile_step is not None:
        compile_step()
    batch_size = arm["device_batch_size"]
    sequence_length = train["sequence_length"]
    tokens = torch.randint(
        0,
        tokenizer.vocab_size,
        (batch_size, sequence_length + 1),
        device=device,
    )
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    loss = compiled(tokens[:, :-1], tokens[:, 1:])
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(tuple(model.parameters()), train["grad_clip"])
    optimizer.step()
    torch.cuda.synchronize(device)
    compiled_step_seconds = time.perf_counter() - started
    loss_value = loss.item()
    grad_norm_value = float(grad_norm)
    if not math.isfinite(loss_value) or not math.isfinite(grad_norm_value):
        raise RuntimeError("compiled baseline step produced non-finite loss or gradients")
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    model.eval()
    short_tokens = tokens[:1, :8]
    with torch.no_grad():
        full_logits = model(short_tokens)
        state = model.state(batch_size=1, length=short_tokens.size(1), device=device)
        cached_logits = torch.cat(
            [
                model(short_tokens[:, index : index + 1], state=state)
                for index in range(short_tokens.size(1))
            ],
            dim=1,
        )
    incremental_error = _maximum_error(cached_logits, full_logits)
    incremental_rtol = 0.02
    incremental_atol = 0.02
    try:
        torch.testing.assert_close(
            cached_logits,
            full_logits,
            rtol=incremental_rtol,
            atol=incremental_atol,
        )
        incremental_passed = True
    except AssertionError:
        incremental_passed = False
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    metadata = {
        "config": model.config.settings(),
        "resolved": {"parameters": model.parameter_count()},
    }
    del (
        cached_logits,
        compiled,
        full_logits,
        grad_norm,
        loss,
        model,
        optimizer,
        short_tokens,
        tokens,
    )
    gc.collect()
    torch.cuda.empty_cache()
    with tempfile.TemporaryDirectory(prefix=f"speck-{arm['id']}-export-") as temporary:
        export = _export_case(state, metadata, Path(temporary))
    del state
    gc.collect()
    torch.cuda.empty_cache()
    hard_peak = 16 * 2**30
    return {
        "arm_id": arm["id"],
        "experiment": str(experiment),
        "config_sha256": {
            f"{name}.json": file_sha256(experiment / f"{name}.json")
            for name in ("model", "tokenizer", "train", "runtime")
        },
        "parameters": arm["parameters"],
        "flops_per_token_at_4096": arm["flops_per_token_at_4096"],
        "compiled_training_step": {
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "loss_backend": train["loss_backend"],
            "optimizer": train["optimizer"],
            "loss": loss_value,
            "gradient_norm_before_clip": grad_norm_value,
            "compile_and_step_seconds": compiled_step_seconds,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "hard_peak_allocated_bytes": hard_peak,
            "within_peak_envelope": peak_allocated <= hard_peak,
        },
        "native_incremental": {
            "tokens": 8,
            "maximum_absolute_logit_error": incremental_error,
            "relative_tolerance": incremental_rtol,
            "absolute_tolerance": incremental_atol,
            "passed": incremental_passed,
        },
        "transformers_export": export,
        "start_temperature_c": start_temperature,
        "thermal_wait_seconds": thermal_wait_seconds,
        "passed": peak_allocated <= hard_peak and incremental_passed and export["passed"],
    }


def run(matrix_path, device, runner_revision):
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise RuntimeError("Paper 1 baseline preflight requires CUDA")
    matrix_path, matrix = load_matrix(matrix_path)
    repository_root = matrix_path.parents[2]
    output_root = repository_root / matrix["planned_primary_baselines"]["output_root"]
    arms = [
        preflight_arm(arm, output_root, torch.device(device))
        for arm in matrix["planned_primary_baselines"]["arms"]
    ]
    return {
        "format": "speck_paper_baseline_preflight",
        "format_version": 1,
        "status": "qualified" if all(arm["passed"] for arm in arms) else "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runner_revision": runner_revision,
        "matrix": str(matrix_path.relative_to(repository_root)),
        "matrix_sha256": file_sha256(matrix_path),
        "hardware": {
            "device": torch.cuda.get_device_name(torch.device(device)),
            "capability": list(torch.cuda.get_device_capability(torch.device(device))),
            "total_memory_bytes": torch.cuda.get_device_properties(
                torch.device(device)
            ).total_memory,
            "driver": command_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
            ).splitlines()[0],
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": importlib.metadata.version("transformers"),
            "flash_linear_attention": importlib.metadata.version("flash-linear-attention"),
        },
        "arms": arms,
        "limitations": [
            "random-weight correctness and feasibility preflight; no quality or throughput claim",
            "one full-size compiled microbatch and optimizer update per arm; not a training benchmark",
            "short native and Transformers incremental-generation parity only",
            "temporary export bytes are hashed in the report but not retained",
        ],
    }


def main(argv=None):
    args = arguments(argv)
    report = run(args.matrix, args.device, repository_revision())
    atomic_json(args.output, report)
    if report["status"] != "qualified":
        raise SystemExit("Paper 1 baseline preflight failed")
    print(f"qualified {len(report['arms'])} Paper 1 baseline arms")


if __name__ == "__main__":
    main()
