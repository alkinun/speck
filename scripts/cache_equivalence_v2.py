"""Evaluate and analyze the control-first Speck CUDA cache-equivalence v2 contract."""

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from scripts.cache_equivalence_cases import token_sha256
from scripts.infer import load_checkpoint_model
from scripts.paper_baseline_preflight import _wait_for_temperature
from speck.checkpoint import checkpoint_identity
from speck.config import load_experiment
from speck.dataloader import packed_loader
from speck.dataset import resolve_data_dir
from speck.paper_baseline import file_sha256
from speck.tokenizer import get_tokenizer


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate", help="evaluate one frozen checkpoint")
    evaluate.add_argument("contract", type=Path)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--control-lock", type=Path, default=None)
    evaluate.add_argument("--output", type=Path, required=True)
    lock = subparsers.add_parser("lock-control", help="lock the dense control before candidates")
    lock.add_argument("contract", type=Path)
    lock.add_argument("control_result", type=Path)
    lock.add_argument("--output", type=Path, required=True)
    analyze = subparsers.add_parser("analyze", help="apply v2 to all candidate checkpoints")
    analyze.add_argument("contract", type=Path)
    analyze.add_argument("control_lock", type=Path)
    analyze.add_argument("candidate_results", nargs="+", type=Path)
    analyze.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def command_output(command, cwd=None):
    result = subprocess.run(command, cwd=cwd, capture_output=True, check=True, text=True)
    return result.stdout.strip()


def repository_revision():
    root = Path(__file__).resolve().parents[1]
    if command_output(["git", "status", "--porcelain"], cwd=root):
        raise ValueError("cache-equivalence v2 requires a clean repository")
    return command_output(["git", "rev-parse", "HEAD"], cwd=root)


def atomic_json(path, value):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_json(path):
    path = Path(path).expanduser().resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load cache-equivalence artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("cache-equivalence artifacts must contain JSON objects")
    return path, value


def load_contract(path):
    path, contract = load_json(path)
    version = contract.get("format_version")
    expected_status = {
        2: "control_calibration_frozen_candidate_unseen",
        3: "powered_prospective_candidate_unseen",
    }.get(version)
    if (
        contract.get("format") != "speck_cache_equivalence_contract"
        or expected_status is None
        or contract.get("status") != expected_status
    ):
        raise ValueError("cache-equivalence contract must use a supported frozen version")
    repository_root = path.parents[2]
    pins = (
        ("case_stream", "path", "sha256"),
        ("checkpoint_source", "path", "sha256"),
    )
    for group, path_key, hash_key in pins:
        artifact = repository_root / contract[group][path_key]
        if not artifact.is_file() or file_sha256(artifact) != contract[group][hash_key]:
            raise ValueError(f"cache-equivalence {group} does not match its pin")
    evidence_pins = (
        (
            ("random_weight_result", "random_weight_result_sha256"),
            ("trained_sentinel_result", "trained_sentinel_result_sha256"),
        )
        if version == 2
        else (("v2_analysis", "v2_analysis_sha256"),)
    )
    for path_key, hash_key in evidence_pins:
        artifact = repository_root / contract["evidence_basis"][path_key]
        if not artifact.is_file() or file_sha256(artifact) != contract["evidence_basis"][hash_key]:
            raise ValueError(f"cache-equivalence evidence {path_key} does not match its pin")
    _, cases = load_json(repository_root / contract["case_stream"]["path"])
    if (
        cases.get("format") != "speck_cache_equivalence_case_stream"
        or cases.get("format_version") != version - 1
        or cases.get("case_stream_sha256") != contract["case_stream"]["stream_sha256"]
        or cases.get("manifest_sha256") != contract["case_stream"]["manifest_sha256"]
    ):
        raise ValueError("cache-equivalence case stream is invalid")
    _, checkpoints = load_json(repository_root / contract["checkpoint_source"]["path"])
    checkpoint_ids = [checkpoint["id"] for checkpoint in checkpoints["checkpoints"]]
    expected = [contract["checkpoint_source"]["control"]] + contract["checkpoint_source"][
        "candidates"
    ]
    if checkpoint_ids != expected:
        raise ValueError("cache-equivalence checkpoint source is invalid")
    statistics_contract = contract["statistics"]
    if (
        statistics_contract.get("paired_resamples") != 10_000
        or statistics_contract.get("one_sided_confidence_level") != 0.95
    ):
        raise ValueError("cache-equivalence statistical contract is invalid")
    if version == 3:
        power_path = repository_root / contract.get("power_analysis", {}).get("path", "")
        if not power_path.is_file() or file_sha256(power_path) != contract["power_analysis"].get(
            "sha256"
        ):
            raise ValueError("cache-equivalence v3 power analysis does not match its pin")
        _, power = load_json(power_path)
        if (
            power.get("format") != "speck_cache_equivalence_power_analysis"
            or power.get("selected_v3_cases_per_length") != 88
            or power.get("free_running_cases_required_if_primary") != 1683
        ):
            raise ValueError("cache-equivalence v3 power analysis is invalid")
    expected_margins = {
        "common_history_argmax_disagreement": ("non_inferiority_margin_absolute", 0.01),
        "high_margin_argmax_disagreement": ("non_inferiority_margin_absolute", 0.002),
        "common_history_js_divergence": ("non_inferiority_margin_absolute", 0.0001),
        "common_history_top10_overlap": ("non_inferiority_margin_absolute", 0.02),
        "common_history_relative_rms_ratio": ("maximum_ratio", 1.5),
        "early_free_running_divergence": ("non_inferiority_margin_absolute", 0.05),
    }
    for endpoint, (field, expected) in expected_margins.items():
        if contract.get("endpoints", {}).get(endpoint, {}).get(field) != expected:
            raise ValueError("cache-equivalence frozen endpoint margins changed")
    if (
        contract.get("execution", {}).get("generation_tokens") != 32
        or contract.get("execution", {}).get("top_k") != 10
        or contract.get("execution", {}).get("margin_thresholds") != [0.1, 0.5]
        or contract.get("case_stream", {}).get("short_cases") != (33 if version == 2 else 88)
        or contract.get("case_stream", {}).get("proxy_4k_cases") != (11 if version == 2 else 88)
    ):
        raise ValueError("cache-equivalence frozen execution geometry changed")
    free_authority = contract["endpoints"]["early_free_running_divergence"].get(
        "authority", "primary"
    )
    if free_authority != ("primary" if version == 2 else "descriptive_risk"):
        raise ValueError("cache-equivalence free-running endpoint authority changed")
    return path, contract, cases, checkpoints


def _case_tensors(contract, cases, repository_root):
    checkpoint_source = load_json(repository_root / contract["checkpoint_source"]["path"])[1]
    source = next(
        checkpoint
        for checkpoint in checkpoint_source["checkpoints"]
        if checkpoint["id"] == contract["checkpoint_source"]["candidates"][0]
    )
    experiment = repository_root / source["experiment"]
    configs = load_experiment(experiment, "data", "tokenizer")
    tokenizer = get_tokenizer(**configs["tokenizer"])
    data_dir = resolve_data_dir(
        configs["data"].get("output_dir"), configs["data"].get("output_name")
    )
    tensors = {}
    records = {}
    for group in cases["groups"]:
        loader = packed_loader(
            tokenizer,
            1,
            group["base_length"],
            "val",
            device="cpu",
            data_dir=data_dir,
        )
        values = []
        for expected_case in group["cases"]:
            inputs, _, state = next(loader)
            observed = {
                "id": expected_case["id"],
                "source": state["selected_source"],
                "global_consumed_tokens": state["global_consumed_tokens"],
                "source_offset": state["source_offsets"][state["selected_source"]],
                "input_sha256": token_sha256(inputs),
                "prefix_sha256": {
                    str(length): token_sha256(inputs[:, :length])
                    for length in group["prefix_lengths"]
                },
            }
            for key, value in observed.items():
                if value != expected_case[key]:
                    raise ValueError(f"cache-equivalence case rematerialization drift: {key}")
            values.append(inputs.squeeze(0))
        stacked = torch.stack(values)
        for length in group["prefix_lengths"]:
            tensors[length] = stacked[:, :length]
            records[length] = [
                {"id": case["id"], "source": case["source"]} for case in group["cases"]
            ]
    return tokenizer, tensors, records


def _distribution_rows(actual, expected, top_k, margin_thresholds):
    actual = actual.float()
    expected = expected.float()
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise FloatingPointError("cache-equivalence logits are non-finite")
    difference = actual - expected
    rms = difference.square().mean(dim=-1).sqrt()
    reference_rms = expected.square().mean(dim=-1).sqrt()
    relative_rms = rms / reference_rms.clamp_min(torch.finfo(torch.float32).tiny)
    actual_logp = torch.log_softmax(actual, dim=-1)
    expected_logp = torch.log_softmax(expected, dim=-1)
    actual_p = actual_logp.exp()
    expected_p = expected_logp.exp()
    mixture = (actual_p + expected_p) * 0.5
    log_mixture = mixture.clamp_min(torch.finfo(torch.float32).tiny).log()
    js = 0.5 * (
        (actual_p * (actual_logp - log_mixture)).sum(dim=-1)
        + (expected_p * (expected_logp - log_mixture)).sum(dim=-1)
    )
    total_variation = 0.5 * (actual_p - expected_p).abs().sum(dim=-1)
    actual_top = actual.topk(top_k, dim=-1).indices
    expected_values, expected_top = expected.topk(top_k, dim=-1)
    overlap = (expected_top[:, :, None] == actual_top[:, None, :]).any(dim=-1).float().mean(dim=-1)
    actual_argmax = actual_top[:, 0]
    expected_argmax = expected_top[:, 0]
    margin = expected_values[:, 0] - expected_values[:, 1]
    rows = []
    for index in range(actual.size(0)):
        disagreement = bool(actual_argmax[index] != expected_argmax[index])
        rows.append(
            {
                "relative_rms": relative_rms[index].item(),
                "jensen_shannon": max(0.0, js[index].item()),
                "total_variation": total_variation[index].item(),
                "top10_overlap": overlap[index].item(),
                "full_margin": margin[index].item(),
                "full_token": expected_argmax[index].item(),
                "cached_token": actual_argmax[index].item(),
                "argmax_disagreement": disagreement,
                "high_margin_disagreement": {
                    str(threshold): disagreement and margin[index].item() >= threshold
                    for threshold in margin_thresholds
                },
            }
        )
    return rows


def compare_batch(model, prompts, steps, top_k, margin_thresholds):
    batch = prompts.size(0)
    state = model.state(batch_size=batch, length=prompts.size(1) + steps, device=prompts.device)
    common_sequence = prompts.clone()
    per_case_steps = [[] for _ in range(batch)]
    with torch.inference_mode():
        cached_logits = model(prompts, state=state, last_token_only=True)[:, -1]
        full_logits = model(common_sequence, last_token_only=True)[:, -1]
        for step in range(steps):
            rows = _distribution_rows(cached_logits, full_logits, top_k, margin_thresholds)
            full_token = full_logits.argmax(dim=-1)
            for index, row in enumerate(rows):
                per_case_steps[index].append({"step": step, **row})
            common_sequence = torch.cat((common_sequence, full_token[:, None]), dim=1)
            cached_logits = model(full_token[:, None], state=state, last_token_only=True)[:, -1]
            full_logits = model(common_sequence, last_token_only=True)[:, -1]

        free_state = model.state(
            batch_size=batch,
            length=prompts.size(1) + steps,
            device=prompts.device,
        )
        free_full_sequence = prompts.clone()
        free_cached_logits = model(prompts, state=free_state, last_token_only=True)[:, -1]
        free_full_logits = model(free_full_sequence, last_token_only=True)[:, -1]
        first_divergence = [None] * batch
        for step in range(steps):
            cached_token = free_cached_logits.argmax(dim=-1)
            full_token = free_full_logits.argmax(dim=-1)
            different = cached_token != full_token
            for index in different.nonzero(as_tuple=False).flatten().tolist():
                if first_divergence[index] is None:
                    first_divergence[index] = step
            free_cached_logits = model(
                cached_token[:, None], state=free_state, last_token_only=True
            )[:, -1]
            free_full_sequence = torch.cat((free_full_sequence, full_token[:, None]), dim=1)
            free_full_logits = model(free_full_sequence, last_token_only=True)[:, -1]
    return per_case_steps, first_divergence


def _quantile(values, q):
    return float(np.quantile(np.asarray(values, dtype=np.float64), q, method="linear"))


def _summarize_cases(cases, margin_thresholds):
    steps = [step for case in cases for step in case["steps"]]
    result = {
        "cases": len(cases),
        "comparisons": len(steps),
        "argmax_disagreement_rate": sum(step["argmax_disagreement"] for step in steps) / len(steps),
        "mean_relative_rms": float(np.mean([step["relative_rms"] for step in steps])),
        "relative_rms_quantiles": {
            str(q): _quantile([step["relative_rms"] for step in steps], q)
            for q in (0.5, 0.9, 0.95, 0.99, 1.0)
        },
        "mean_jensen_shannon": float(np.mean([step["jensen_shannon"] for step in steps])),
        "jensen_shannon_quantiles": {
            str(q): _quantile([step["jensen_shannon"] for step in steps], q)
            for q in (0.5, 0.9, 0.95, 0.99, 1.0)
        },
        "mean_total_variation": float(np.mean([step["total_variation"] for step in steps])),
        "mean_top10_overlap": float(np.mean([step["top10_overlap"] for step in steps])),
        "top10_overlap_quantiles": {
            str(q): _quantile([step["top10_overlap"] for step in steps], q)
            for q in (0.0, 0.01, 0.05, 0.5)
        },
        "common_history_case_divergence_rate": sum(
            any(step["argmax_disagreement"] for step in case["steps"]) for case in cases
        )
        / len(cases),
        "early_free_running_divergence_rate": sum(
            case["free_running_first_divergence_step"] is not None for case in cases
        )
        / len(cases),
    }
    result["high_margin"] = {}
    for threshold in margin_thresholds:
        eligible = [step for step in steps if step["full_margin"] >= threshold]
        disagreements = [
            step for step in eligible if step["high_margin_disagreement"][str(threshold)]
        ]
        result["high_margin"][str(threshold)] = {
            "comparisons": len(eligible),
            "disagreements": len(disagreements),
            "disagreement_rate": len(disagreements) / len(eligible) if eligible else None,
        }
    return result


def _checkpoint_map(contract, checkpoints):
    return {checkpoint["id"]: checkpoint for checkpoint in checkpoints["checkpoints"]}


def _validate_lock(contract_path, contract, lock_path):
    lock_path, lock = load_json(lock_path)
    if (
        lock.get("format") != "speck_cache_equivalence_control_lock"
        or lock.get("format_version") != contract["format_version"]
        or lock.get("status") != "dense_control_locked_before_candidate_execution"
        or lock.get("contract_sha256") != file_sha256(contract_path)
        or lock.get("control_checkpoint") != contract["checkpoint_source"]["control"]
    ):
        raise ValueError("cache-equivalence control lock is invalid")
    return lock_path, lock


def evaluate(contract_path, checkpoint_id, runner_revision, control_lock_path=None):
    contract_path, contract, cases, checkpoints = load_contract(contract_path)
    repository_root = contract_path.parents[2]
    checkpoint_map = _checkpoint_map(contract, checkpoints)
    if checkpoint_id not in checkpoint_map:
        raise ValueError("cache-equivalence checkpoint is outside the frozen contract")
    control_id = contract["checkpoint_source"]["control"]
    lock_reference = None
    if checkpoint_id == control_id:
        if control_lock_path is not None:
            raise ValueError("dense control must execute before a control lock exists")
    else:
        if control_lock_path is None:
            raise ValueError("candidate execution requires the frozen dense control lock")
        lock_path, lock = _validate_lock(contract_path, contract, control_lock_path)
        lock_reference = {"path": str(lock_path), "sha256": file_sha256(lock_path)}
    checkpoint = checkpoint_map[checkpoint_id]
    tokenizer, tensors, records = _case_tensors(contract, cases, repository_root)
    experiment = repository_root / checkpoint["experiment"]
    source_checkpoint = checkpoint_map[contract["checkpoint_source"]["candidates"][0]]
    source_experiment = repository_root / source_checkpoint["experiment"]
    if (
        load_experiment(experiment, "tokenizer")["tokenizer"]
        != load_experiment(source_experiment, "tokenizer")["tokenizer"]
    ):
        raise ValueError("cache-equivalence checkpoint tokenizer mismatch")
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
        raise ValueError("cache-equivalence checkpoint identity changed")
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
        raise ValueError("cache-equivalence checkpoint metadata changed")
    execution = contract["execution"]
    all_results = []
    for length in sorted(tensors):
        length_cases = []
        batch_size = execution["batch_sizes"][str(length)]
        for start in range(0, tensors[length].size(0), batch_size):
            prompt = tensors[length][start : start + batch_size].to("cuda")
            steps, free_divergence = compare_batch(
                model,
                prompt,
                execution["generation_tokens"],
                execution["top_k"],
                execution["margin_thresholds"],
            )
            for offset, case_steps in enumerate(steps):
                case = records[length][start + offset]
                length_cases.append(
                    {
                        **case,
                        "steps": case_steps,
                        "free_running_first_divergence_step": free_divergence[offset],
                    }
                )
        all_results.append(
            {
                "prompt_tokens": length,
                "cases": length_cases,
                "summary": _summarize_cases(length_cases, execution["margin_thresholds"]),
            }
        )
    peak_allocated = torch.cuda.max_memory_allocated()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "format": "speck_cache_equivalence_checkpoint_result",
        "format_version": contract["format_version"],
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runner_revision": runner_revision,
        "contract": str(contract_path.relative_to(repository_root)),
        "contract_sha256": file_sha256(contract_path),
        "case_stream_sha256": contract["case_stream"]["stream_sha256"],
        "checkpoint_id": checkpoint_id,
        "role": "control" if checkpoint_id == control_id else "candidate",
        "checkpoint": identity,
        "control_lock": lock_reference,
        "start_temperature_c": start_temperature,
        "thermal_wait_seconds": thermal_wait_seconds,
        "peak_allocated_bytes": peak_allocated,
        "results": all_results,
        "hardware": {
            "device": torch.cuda.get_device_name(),
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
    }


def lock_control(contract_path, control_result_path):
    contract_path, contract, _, _ = load_contract(contract_path)
    control_result_path, control = load_json(control_result_path)
    if (
        control.get("format") != "speck_cache_equivalence_checkpoint_result"
        or control.get("format_version") != contract["format_version"]
        or control.get("status") != "complete"
        or control.get("role") != "control"
        or control.get("checkpoint_id") != contract["checkpoint_source"]["control"]
        or control.get("contract_sha256") != file_sha256(contract_path)
    ):
        raise ValueError("cache-equivalence dense control result is invalid")
    return {
        "format": "speck_cache_equivalence_control_lock",
        "format_version": contract["format_version"],
        "status": "dense_control_locked_before_candidate_execution",
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": file_sha256(contract_path),
        "control_checkpoint": control["checkpoint_id"],
        "control_result": {
            "path": str(control_result_path),
            "sha256": file_sha256(control_result_path),
        },
        "endpoints": contract["endpoints"],
        "statistics": contract["statistics"],
    }


def _case_endpoint(case, endpoint):
    steps = case["steps"]
    if endpoint == "argmax":
        return sum(step["argmax_disagreement"] for step in steps) / len(steps)
    if endpoint == "high_margin":
        eligible = [step for step in steps if step["full_margin"] >= 0.1]
        if not eligible:
            return 0.0
        return sum(step["argmax_disagreement"] for step in eligible) / len(eligible)
    if endpoint == "js":
        return float(np.mean([step["jensen_shannon"] for step in steps]))
    if endpoint == "top10":
        return float(np.mean([step["top10_overlap"] for step in steps]))
    if endpoint == "relative_rms":
        return float(np.mean([step["relative_rms"] for step in steps]))
    if endpoint == "free":
        return float(case["free_running_first_divergence_step"] is not None)
    raise ValueError(f"unknown cache-equivalence endpoint: {endpoint}")


def _paired_bootstrap(control_cases, candidate_cases, endpoint, contract, seed_offset):
    control_by_id = {case["id"]: case for case in control_cases}
    candidate_by_id = {case["id"]: case for case in candidate_cases}
    if set(control_by_id) != set(candidate_by_id):
        raise ValueError("cache-equivalence paired case ids do not match")
    ids = sorted(control_by_id)
    control = np.asarray([_case_endpoint(control_by_id[id], endpoint) for id in ids])
    candidate = np.asarray([_case_endpoint(candidate_by_id[id], endpoint) for id in ids])
    rng = np.random.default_rng(contract["statistics"]["bootstrap_seed"] + seed_offset)
    indices = rng.integers(0, len(ids), size=(contract["statistics"]["paired_resamples"], len(ids)))
    control_means = control[indices].mean(axis=1)
    candidate_means = candidate[indices].mean(axis=1)
    differences = candidate_means - control_means
    return {
        "cases": len(ids),
        "control_mean": float(control.mean()),
        "candidate_mean": float(candidate.mean()),
        "candidate_minus_control": float(candidate.mean() - control.mean()),
        "upper_one_sided_95_bound": float(np.quantile(differences, 0.95)),
        "lower_one_sided_95_bound": float(np.quantile(differences, 0.05)),
        "ratio": float(candidate.mean() / control.mean()) if control.mean() else None,
        "ratio_upper_one_sided_95_bound": (
            float(
                np.quantile(
                    candidate_means[control_means > 0] / control_means[control_means > 0],
                    0.95,
                )
            )
            if np.any(control_means > 0)
            else None
        ),
    }


def analyze(contract_path, control_lock_path, candidate_result_paths):
    contract_path, contract, _, _ = load_contract(contract_path)
    lock_path, lock = _validate_lock(contract_path, contract, control_lock_path)
    control_path = Path(lock["control_result"]["path"])
    if file_sha256(control_path) != lock["control_result"]["sha256"]:
        raise ValueError("cache-equivalence locked dense control changed")
    _, control = load_json(control_path)
    candidates = []
    references = []
    for path in candidate_result_paths:
        path, result = load_json(path)
        if (
            result.get("format") != "speck_cache_equivalence_checkpoint_result"
            or result.get("format_version") != contract["format_version"]
            or result.get("status") != "complete"
            or result.get("role") != "candidate"
            or result.get("contract_sha256") != file_sha256(contract_path)
            or result.get("created_at") <= lock["locked_at"]
        ):
            raise ValueError("cache-equivalence candidate result is invalid")
        candidates.append(result)
        references.append({"path": str(path), "sha256": file_sha256(path)})
    expected_ids = contract["checkpoint_source"]["candidates"]
    if sorted(result["checkpoint_id"] for result in candidates) != sorted(expected_ids):
        raise ValueError("cache-equivalence candidate results are incomplete")
    control_by_length = {result["prompt_tokens"]: result for result in control["results"]}
    endpoint_rules = {
        "argmax": ("common_history_argmax_disagreement", "upper"),
        "high_margin": ("high_margin_argmax_disagreement", "upper"),
        "js": ("common_history_js_divergence", "upper"),
        "top10": ("common_history_top10_overlap", "lower"),
        "relative_rms": ("common_history_relative_rms_ratio", "ratio"),
        "free": ("early_free_running_divergence", "upper"),
    }
    if contract["format_version"] == 3:
        endpoint_rules["free"] = ("early_free_running_divergence", "descriptive")
    decisions = []
    for candidate_index, candidate in enumerate(candidates):
        candidate_by_length = {result["prompt_tokens"]: result for result in candidate["results"]}
        length_results = []
        for length_index, length in enumerate(sorted(control_by_length)):
            endpoints = {}
            for endpoint_index, (endpoint, (rule_name, direction)) in enumerate(
                endpoint_rules.items()
            ):
                bootstrap = _paired_bootstrap(
                    control_by_length[length]["cases"],
                    candidate_by_length[length]["cases"],
                    endpoint,
                    contract,
                    candidate_index * 100 + length_index * 10 + endpoint_index,
                )
                rule = contract["endpoints"][rule_name]
                if direction == "upper":
                    passed = (
                        bootstrap["upper_one_sided_95_bound"]
                        <= rule["non_inferiority_margin_absolute"]
                    )
                elif direction == "lower":
                    passed = (
                        bootstrap["lower_one_sided_95_bound"]
                        >= -rule["non_inferiority_margin_absolute"]
                    )
                elif direction == "ratio":
                    bound = bootstrap["ratio_upper_one_sided_95_bound"]
                    passed = bound is not None and bound <= rule["maximum_ratio"]
                else:
                    passed = None
                endpoints[rule_name] = {**bootstrap, "passed": passed}
            high_margin_half = candidate_by_length[length]["summary"]["high_margin"]["0.5"]
            hard_guardrail = high_margin_half["disagreements"] == 0
            length_results.append(
                {
                    "prompt_tokens": length,
                    "endpoints": endpoints,
                    "high_margin_0.5_guardrail_pass": hard_guardrail,
                    "passed": hard_guardrail
                    and all(
                        result["passed"]
                        for result in endpoints.values()
                        if result["passed"] is not None
                    ),
                }
            )
        decisions.append(
            {
                "checkpoint_id": candidate["checkpoint_id"],
                "lengths": length_results,
                "passed": all(result["passed"] for result in length_results),
            }
        )
    return {
        "format": "speck_cache_equivalence_analysis",
        "format_version": contract["format_version"],
        "status": "qualified" if all(result["passed"] for result in decisions) else "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": file_sha256(contract_path),
        "control_lock": {"path": str(lock_path), "sha256": file_sha256(lock_path)},
        "candidate_results": references,
        "decisions": decisions,
        "authority": contract["decision_scope"],
    }


def run(args, runner_revision):
    if args.command == "evaluate":
        return evaluate(
            args.contract,
            args.checkpoint,
            runner_revision,
            control_lock_path=args.control_lock,
        )
    if args.command == "lock-control":
        return lock_control(args.contract, args.control_result)
    return analyze(args.contract, args.control_lock, args.candidate_results)


def main(argv=None):
    args = arguments(argv)
    result = run(args, repository_revision())
    atomic_json(args.output, result)
    print(f"{result['format']}: {result['status']}")


if __name__ == "__main__":
    main()
