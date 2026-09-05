"""Run trained-checkpoint CUDA decode sentinels after the Paper 1 random-weight matrix."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from scripts.cuda_decode_diagnostic import compare_full_and_cached, logit_metrics
from scripts.infer import load_checkpoint_model
from scripts.paper_baseline_preflight import _wait_for_temperature
from speck.checkpoint import checkpoint_identity
from speck.config import load_experiment
from speck.dataloader import manifest_fingerprint, packed_loader
from speck.dataset import load_manifest, resolve_data_dir
from speck.paper_baseline import file_sha256
from speck.tokenizer import get_tokenizer


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def command_output(command, cwd=None):
    result = subprocess.run(command, cwd=cwd, capture_output=True, check=True, text=True)
    return result.stdout.strip()


def repository_revision():
    root = Path(__file__).resolve().parents[1]
    if command_output(["git", "status", "--porcelain"], cwd=root):
        raise ValueError("trained decode sentinel requires a clean repository")
    return command_output(["git", "rev-parse", "HEAD"], cwd=root)


def atomic_json(path, value):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_contract(path):
    path = Path(path).expanduser().resolve()
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load trained decode sentinel contract: {error}") from error
    if (
        contract.get("format") != "speck_cuda_decode_trained_sentinel_contract"
        or contract.get("format_version") != 1
        or contract.get("status") != "frozen_after_random_weight_matrix_before_trained_execution"
    ):
        raise ValueError("trained decode sentinel contract must use frozen format version 1")
    repository_root = path.parents[2]
    trigger = repository_root / contract.get("trigger_result", "")
    if not trigger.is_file() or file_sha256(trigger) != contract.get("trigger_result_sha256"):
        raise ValueError("trained decode sentinel trigger result does not match its pin")
    data = contract.get("data", {})
    if data.get("prefix_lengths") != [8, 64, 512] or contract.get("greedy_generation_tokens") != 32:
        raise ValueError("trained decode sentinel data/generation geometry is invalid")
    checkpoints = contract.get("checkpoints", ())
    if [checkpoint.get("id") for checkpoint in checkpoints] != [
        "dense_global_seed42",
        "kda_seed42",
        "kda_seed43",
        "kda_seed44",
    ]:
        raise ValueError("trained decode sentinel checkpoint matrix is invalid")
    numerical = contract.get("numerical_contract", {})
    if numerical.get("relative_tolerance") != 0.02 or numerical.get("absolute_tolerance") != 0.02:
        raise ValueError("trained decode sentinel changed the frozen tolerance")
    return path, contract


def validation_tokens(contract, repository_root):
    experiment = repository_root / contract["data"]["source_experiment"]
    configs = load_experiment(experiment, "data", "tokenizer")
    tokenizer = get_tokenizer(**configs["tokenizer"])
    data_dir = resolve_data_dir(
        configs["data"].get("output_dir"), configs["data"].get("output_name")
    )
    manifest = load_manifest(data_dir)
    observed_manifest = manifest_fingerprint(manifest)
    if observed_manifest != contract["data"]["manifest_sha256"]:
        raise ValueError("trained decode sentinel packed-data manifest does not match")
    loader = packed_loader(
        tokenizer,
        1,
        max(contract["data"]["prefix_lengths"]),
        contract["data"]["split"],
        device="cpu",
        data_dir=data_dir,
    )
    inputs, _, _ = next(loader)
    digest = hashlib.sha256(inputs.numpy().tobytes()).hexdigest()
    return tokenizer, inputs, str(data_dir), digest


def compare_common_history(model, prompt, steps, rtol, atol):
    state = model.state(
        batch_size=prompt.size(0),
        length=prompt.size(1) + steps,
        device=prompt.device,
    )
    sequence = prompt.clone()
    with torch.inference_mode():
        cached_logits = model(prompt, state=state, last_token_only=True)[:, -1]
        recompute_logits = model(sequence, last_token_only=True)[:, -1]
        records = []
        for step in range(steps):
            metrics = logit_metrics(cached_logits[:, None], recompute_logits[:, None], rtol, atol)
            recompute_token = recompute_logits.argmax(dim=-1)
            cached_token = cached_logits.argmax(dim=-1)
            records.append(
                {
                    "step": step,
                    "recompute_token": recompute_token.item(),
                    "cached_token": cached_token.item(),
                    **metrics,
                }
            )
            sequence = torch.cat((sequence, recompute_token[:, None]), dim=1)
            cached_logits = model(recompute_token[:, None], state=state, last_token_only=True)[
                :, -1
            ]
            recompute_logits = model(sequence, last_token_only=True)[:, -1]
    return {
        "steps": records,
        "argmax_agreement": sum(
            record["recompute_token"] == record["cached_token"] for record in records
        )
        / len(records),
        "first_argmax_disagreement_step": next(
            (
                record["step"]
                for record in records
                if record["recompute_token"] != record["cached_token"]
            ),
            None,
        ),
    }


def compare_free_running(model, prompt, steps):
    state = model.state(
        batch_size=prompt.size(0),
        length=prompt.size(1) + steps,
        device=prompt.device,
    )
    recompute_sequence = prompt.clone()
    cached_tokens = []
    recompute_tokens = []
    first_disagreement = None
    with torch.inference_mode():
        cached_logits = model(prompt, state=state, last_token_only=True)[:, -1]
        recompute_logits = model(recompute_sequence, last_token_only=True)[:, -1]
        for step in range(steps):
            cached_token = cached_logits.argmax(dim=-1)
            recompute_token = recompute_logits.argmax(dim=-1)
            cached_tokens.append(cached_token.item())
            recompute_tokens.append(recompute_token.item())
            if first_disagreement is None and not torch.equal(cached_token, recompute_token):
                first_disagreement = step
            cached_logits = model(cached_token[:, None], state=state, last_token_only=True)[:, -1]
            recompute_sequence = torch.cat((recompute_sequence, recompute_token[:, None]), dim=1)
            recompute_logits = model(recompute_sequence, last_token_only=True)[:, -1]
    return {
        "cached_tokens": cached_tokens,
        "recompute_tokens": recompute_tokens,
        "first_argmax_disagreement_step": first_disagreement,
        "complete_sequence_match": cached_tokens == recompute_tokens,
    }


def run_checkpoint(checkpoint, contract, repository_root, tokens, tokenizer):
    experiment = repository_root / checkpoint["experiment"]
    configs = load_experiment(experiment, "tokenizer")
    if (
        configs["tokenizer"]
        != load_experiment(repository_root / contract["data"]["source_experiment"], "tokenizer")[
            "tokenizer"
        ]
    ):
        raise ValueError("trained decode sentinel tokenizer mismatch")
    checkpoint_dir = Path.home() / ".cache" / "speck" / "checkpoints" / checkpoint["run"]
    identity = checkpoint_identity(checkpoint_dir, checkpoint["step"])
    expected_identity = {
        "directory": str(checkpoint_dir),
        "step": checkpoint["step"],
        "model_sha256": checkpoint["model_sha256"],
        "optimizer_sha256": checkpoint["optimizer_sha256"],
        "metadata_sha256": checkpoint["metadata_sha256"],
    }
    if identity != expected_identity:
        raise ValueError(f"trained decode checkpoint identity changed: {checkpoint['id']}")
    start_temperature, thermal_wait_seconds = _wait_for_temperature()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, metadata = load_checkpoint_model(checkpoint_dir, checkpoint["step"], "cuda")
    if (
        metadata["resolved"]["parameters"] != checkpoint["parameters"]
        or metadata["resolved"]["seed"] != checkpoint["seed"]
        or metadata["validation_loss"] != checkpoint["validation_loss"]
        or model.config.vocab_size != tokenizer.vocab_size
    ):
        raise ValueError(f"trained decode checkpoint metadata changed: {checkpoint['id']}")
    numerical = contract["numerical_contract"]
    length_results = []
    for length in contract["data"]["prefix_lengths"]:
        prompt = tokens[:, :length].to("cuda")
        teacher_forced = compare_full_and_cached(
            model,
            prompt,
            numerical["relative_tolerance"],
            numerical["absolute_tolerance"],
        )
        common = compare_common_history(
            model,
            prompt,
            contract["greedy_generation_tokens"],
            numerical["relative_tolerance"],
            numerical["absolute_tolerance"],
        )
        free = compare_free_running(model, prompt, contract["greedy_generation_tokens"])
        length_results.append(
            {
                "prompt_tokens": length,
                "teacher_forced": teacher_forced,
                "common_history_generation": common,
                "free_running_generation": free,
            }
        )
    peak_allocated = torch.cuda.max_memory_allocated()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "id": checkpoint["id"],
        "arm": checkpoint["arm"],
        "seed": checkpoint["seed"],
        "checkpoint": identity,
        "experiment": checkpoint["experiment"],
        "start_temperature_c": start_temperature,
        "thermal_wait_seconds": thermal_wait_seconds,
        "peak_allocated_bytes": peak_allocated,
        "lengths": length_results,
    }


def summarize(results):
    summary = []
    for result in results:
        for length in result["lengths"]:
            teacher = length["teacher_forced"]["logits"]
            common = length["common_history_generation"]
            free = length["free_running_generation"]
            summary.append(
                {
                    "id": result["id"],
                    "arm": result["arm"],
                    "seed": result["seed"],
                    "prompt_tokens": length["prompt_tokens"],
                    "teacher_forced_relative_rms_error": teacher["relative_rms_error"],
                    "teacher_forced_argmax_agreement": teacher["argmax_agreement"],
                    "teacher_forced_final_token_argmax_agreement": teacher[
                        "final_token_argmax_agreement"
                    ],
                    "common_history_argmax_agreement": common["argmax_agreement"],
                    "common_history_first_disagreement_step": common[
                        "first_argmax_disagreement_step"
                    ],
                    "free_running_first_disagreement_step": free["first_argmax_disagreement_step"],
                    "free_running_complete_sequence_match": free["complete_sequence_match"],
                }
            )
    return summary


def run(contract_path, runner_revision):
    contract_path, contract = load_contract(contract_path)
    if not torch.cuda.is_available():
        raise RuntimeError("trained decode sentinel requires CUDA")
    repository_root = contract_path.parents[2]
    tokenizer, tokens, data_dir, tokens_sha256 = validation_tokens(contract, repository_root)
    results = [
        run_checkpoint(checkpoint, contract, repository_root, tokens, tokenizer)
        for checkpoint in contract["checkpoints"]
    ]
    return {
        "format": "speck_cuda_decode_trained_sentinel",
        "format_version": 1,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runner_revision": runner_revision,
        "contract": str(contract_path.relative_to(repository_root)),
        "contract_sha256": file_sha256(contract_path),
        "data": {
            "directory": data_dir,
            "manifest_sha256": contract["data"]["manifest_sha256"],
            "tokens_sha256": tokens_sha256,
            "tokens": tokens.size(1),
        },
        "hardware": {
            "device": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "driver": command_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
            ).splitlines()[0],
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "flash_linear_attention": importlib.metadata.version("flash-linear-attention"),
        },
        "results": results,
        "summary": summarize(results),
        "decision_scope": contract["decision_scope"],
    }


def main(argv=None):
    args = arguments(argv)
    report = run(args.contract, repository_revision())
    atomic_json(args.output, report)
    print(f"completed {len(report['results'])} trained CUDA decode sentinels")


if __name__ == "__main__":
    main()
