"""Run the frozen full-depth CUDA decode diagnostic for Speck Paper 1."""

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

import speck.model as model_module
from scripts.paper_baseline_preflight import _wait_for_temperature
from speck.config import load_experiment
from speck.model import Linear, Operation, RMSNorm, Stage, build_model
from speck.paper_baseline import file_sha256, load_matrix
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
        raise ValueError("CUDA decode diagnostic requires a clean repository")
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
        raise ValueError(f"cannot load CUDA decode diagnostic contract: {error}") from error
    if (
        contract.get("format") != "speck_cuda_decode_diagnostic_contract"
        or contract.get("format_version") != 1
        or contract.get("status") != "frozen_after_length8_pilot_before_matrix_execution"
    ):
        raise ValueError("CUDA decode diagnostic contract must use frozen format version 1")
    repository_root = path.parents[2]
    matrix_path = repository_root / contract.get("baseline_matrix", "")
    trigger_path = repository_root / contract.get("trigger_result", "")
    if not matrix_path.is_file() or file_sha256(matrix_path) != contract.get(
        "baseline_matrix_sha256"
    ):
        raise ValueError("CUDA decode diagnostic baseline matrix does not match its pin")
    if not trigger_path.is_file() or file_sha256(trigger_path) != contract.get(
        "trigger_result_sha256"
    ):
        raise ValueError("CUDA decode diagnostic trigger result does not match its pin")
    _, matrix = load_matrix(matrix_path)
    arm_ids = {arm["id"] for arm in matrix["planned_primary_baselines"]["arms"]}
    if set(contract.get("arms", ())) != arm_ids:
        raise ValueError("CUDA decode diagnostic arms do not match the baseline matrix")
    if contract.get("seeds") != [42, 43, 44] or contract.get("lengths") != [8, 64, 512]:
        raise ValueError("CUDA decode diagnostic seed/length matrix is invalid")
    execution_ids = [value.get("id") for value in contract.get("execution_matrices", ())]
    if execution_ids != [
        "native_cuda",
        "torch_kda_cuda",
        "torch_kda_no_conv_history_cuda",
        "cpu_sentinel",
    ]:
        raise ValueError("CUDA decode diagnostic execution matrices are invalid")
    numerical = contract.get("numerical_contract", {})
    if numerical.get("relative_tolerance") != 0.02 or numerical.get("absolute_tolerance") != 0.02:
        raise ValueError("CUDA decode diagnostic changed the frozen tolerance")
    return path, contract, matrix_path, matrix


def tensor_metrics(actual, expected, rtol=0.02, atol=0.02):
    actual = actual.float()
    expected = expected.float()
    difference = (actual - expected).abs()
    allowed = atol + rtol * expected.abs()
    mismatched = difference > allowed
    rms = difference.square().mean().sqrt().item()
    reference_rms = expected.square().mean().sqrt().item()
    actual_norm = torch.linalg.vector_norm(actual).item()
    reference_norm = torch.linalg.vector_norm(expected).item()
    flattened_actual = actual.flatten(1)
    flattened_expected = expected.flatten(1)
    cosine = F.cosine_similarity(flattened_actual, flattened_expected, dim=-1).mean().item()
    return {
        "maximum_absolute_error": difference.max().item(),
        "rms_error": rms,
        "relative_rms_error": rms / reference_rms if reference_rms else None,
        "mismatched_elements": mismatched.sum().item(),
        "total_elements": mismatched.numel(),
        "cosine_similarity": cosine,
        "actual_norm": actual_norm,
        "reference_norm": reference_norm,
        "norm_ratio": actual_norm / reference_norm if reference_norm else None,
        "relative_tolerance": rtol,
        "absolute_tolerance": atol,
        "passed": not mismatched.any().item(),
    }


def logit_metrics(actual, expected, rtol=0.02, atol=0.02):
    metrics = tensor_metrics(actual, expected, rtol, atol)
    actual_tokens = actual.float().argmax(dim=-1)
    expected_tokens = expected.float().argmax(dim=-1)
    disagreements = (actual_tokens != expected_tokens).nonzero(as_tuple=False)
    metrics.update(
        {
            "argmax_agreement": (actual_tokens == expected_tokens).float().mean().item(),
            "argmax_disagreements": disagreements.size(0),
            "argmax_disagreement_positions": disagreements[:, -1].tolist(),
            "final_token_argmax_agreement": bool(
                torch.equal(actual_tokens[:, -1], expected_tokens[:, -1])
            ),
        }
    )
    return metrics


def _module_description(name, module):
    if isinstance(module, Stage):
        return {"name": name, "type": "stage_residual", "stage_index": module.stage_index}
    if isinstance(module, Linear):
        return {"name": name, "type": "linear"}
    if isinstance(module, RMSNorm):
        return {"name": name, "type": "rms_norm"}
    spec = module.spec
    result = {"name": name, "type": "operation", "kind": spec.kind}
    if spec.kind == "attention":
        result["scope"] = spec.scope
    return result


def _last_sequence_output(name, output):
    if output.ndim == 4 and name.endswith((".q_norm", ".k_norm")):
        return output[:, :, -1]
    return output[:, -1]


def compare_full_and_cached(model, tokens, rtol, atol):
    descriptions = {}
    full_outputs = {}
    cached_outputs = {}
    execution_order = []
    capture = "full"
    capture_cached = False
    hooks = []

    def hook(name, module, output):
        nonlocal capture, capture_cached
        if capture == "full":
            if name in full_outputs:
                raise RuntimeError(f"diagnostic module executed twice in full mode: {name}")
            execution_order.append(name)
            descriptions[name] = _module_description(name, module)
            full_outputs[name] = _last_sequence_output(name, output).detach().cpu()
        elif capture_cached:
            cached_outputs[name] = _last_sequence_output(name, output).detach().cpu()

    for name, module in model.named_modules():
        if isinstance(module, (Linear, RMSNorm, Operation, Stage)):
            hooks.append(
                module.register_forward_hook(
                    lambda module, inputs, output, name=name: hook(name, module, output)
                )
            )
    try:
        with torch.inference_mode():
            started = time.perf_counter()
            full_logits = model(tokens)
            if tokens.is_cuda:
                torch.cuda.synchronize(tokens.device)
            full_seconds = time.perf_counter() - started
            capture = "cached"
            state = model.state(
                batch_size=tokens.size(0),
                length=tokens.size(1),
                device=tokens.device,
            )
            cached = []
            started = time.perf_counter()
            for index in range(tokens.size(1)):
                capture_cached = index + 1 == tokens.size(1)
                cached.append(model(tokens[:, index : index + 1], state=state))
            cached_logits = torch.cat(cached, dim=1)
            if tokens.is_cuda:
                torch.cuda.synchronize(tokens.device)
            cached_seconds = time.perf_counter() - started
    finally:
        for handle in hooks:
            handle.remove()
    if set(full_outputs) != set(cached_outputs):
        raise RuntimeError("diagnostic full and cached module captures do not match")
    modules = []
    for index, name in enumerate(execution_order):
        modules.append(
            {
                "execution_index": index,
                **descriptions[name],
                **tensor_metrics(cached_outputs[name], full_outputs[name], rtol, atol),
            }
        )
    logits = logit_metrics(cached_logits, full_logits, rtol, atol)
    first_failure = next(
        (module for module in modules if module["mismatched_elements"]),
        None,
    )
    first_operation = next(module for module in modules if module["type"] == "operation")
    initial_relative = first_operation["relative_rms_error"]
    final_relative = logits["relative_rms_error"]
    return {
        "tokens": tokens.size(1),
        "full_seconds": full_seconds,
        "cached_seconds": cached_seconds,
        "logits": logits,
        "first_failure": (
            {
                "execution_index": first_failure["execution_index"],
                "name": first_failure["name"],
                "type": first_failure["type"],
                "kind": first_failure.get("kind"),
                "mismatched_elements": first_failure["mismatched_elements"],
            }
            if first_failure
            else None
        ),
        "first_operation_relative_rms_error": initial_relative,
        "final_to_first_relative_rms_amplification": (
            final_relative / initial_relative if initial_relative else None
        ),
        "modules": modules,
    }


@contextmanager
def kda_backend(mode):
    original = model_module.kimi_delta_rule
    if mode in {"torch_kda_cuda", "torch_kda_no_conv_history_cuda"}:
        model_module.kimi_delta_rule = model_module.torch_kimi_delta_rule
    try:
        yield
    finally:
        model_module.kimi_delta_rule = original


def _tokens(vocab_size, seed, maximum_length, device):
    generator = torch.Generator().manual_seed(seed + 10_000)
    return torch.randint(0, vocab_size, (1, maximum_length), generator=generator).to(device)


def run_configuration(configuration, arm, output_root, rtol, atol):
    experiment = output_root / "arms" / arm["id"]
    configs = load_experiment(experiment, "model", "tokenizer")
    tokenizer = get_tokenizer(**configs["tokenizer"])
    device = torch.device(configuration["device"])
    cases = []
    for seed in configuration["seeds"]:
        if device.type == "cuda":
            start_temperature, thermal_wait_seconds = _wait_for_temperature()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.manual_seed(seed)
        else:
            start_temperature, thermal_wait_seconds = None, 0.0
        torch.manual_seed(seed)
        with kda_backend(configuration["id"]):
            model = build_model(
                configs["model"],
                tokenizer.vocab_size,
                tokenizer.bos_id,
                tokenizer.eos_id,
            ).to(device)
            model.init_weights()
            if configuration["id"] == "torch_kda_no_conv_history_cuda":
                with torch.no_grad():
                    for name, parameter in model.named_parameters():
                        if name.endswith("conv_kernel"):
                            parameter[..., :-1].zero_()
            model.eval()
            tokens = _tokens(
                tokenizer.vocab_size,
                seed,
                max(configuration["lengths"]),
                device,
            )
            lengths = []
            for length in configuration["lengths"]:
                lengths.append(compare_full_and_cached(model, tokens[:, :length], rtol, atol))
            peak_allocated = (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            )
        cases.append(
            {
                "seed": seed,
                "start_temperature_c": start_temperature,
                "thermal_wait_seconds": thermal_wait_seconds,
                "peak_allocated_bytes": peak_allocated,
                "lengths": lengths,
            }
        )
        del model, tokens
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "execution_matrix": configuration["id"],
        "arm_id": arm["id"],
        "device": str(device),
        "parameters": arm["parameters"],
        "model_sha256": file_sha256(experiment / "model.json"),
        "cases": cases,
    }


def _case_index(results):
    return {
        (result["execution_matrix"], result["arm_id"], case["seed"], length["tokens"]): length
        for result in results
        for case in result["cases"]
        for length in case["lengths"]
    }


def summarize(results, contract):
    index = _case_index(results)
    native = []
    torch_kda = []
    no_conv_history = []
    for arm in contract["arms"]:
        for length in contract["lengths"]:
            cells = [
                index[("native_cuda", arm, seed, length)]["logits"] for seed in contract["seeds"]
            ]
            native.append(
                {
                    "arm_id": arm,
                    "tokens": length,
                    "relative_rms_error_median": statistics.median(
                        cell["relative_rms_error"] for cell in cells
                    ),
                    "relative_rms_error_range": [
                        min(cell["relative_rms_error"] for cell in cells),
                        max(cell["relative_rms_error"] for cell in cells),
                    ],
                    "argmax_agreement_median": statistics.median(
                        cell["argmax_agreement"] for cell in cells
                    ),
                    "all_cells_pass": all(cell["passed"] for cell in cells),
                }
            )
    sole_cause_falsified = []
    kda_arm = "five_cache_kda_gqa"
    for seed in contract["seeds"]:
        for length in contract["lengths"]:
            native_cell = index[("native_cuda", kda_arm, seed, length)]["logits"]
            torch_cell = index[("torch_kda_cuda", kda_arm, seed, length)]["logits"]
            retained_fraction = (
                torch_cell["relative_rms_error"] / native_cell["relative_rms_error"]
                if native_cell["relative_rms_error"]
                else None
            )
            falsified = bool(torch_cell["argmax_disagreements"]) or (
                retained_fraction is not None and retained_fraction >= 0.5
            )
            cell = {
                "seed": seed,
                "tokens": length,
                "native_relative_rms_error": native_cell["relative_rms_error"],
                "torch_relative_rms_error": torch_cell["relative_rms_error"],
                "torch_over_native_relative_rms": retained_fraction,
                "torch_argmax_disagreements": torch_cell["argmax_disagreements"],
                "fla_sole_cause_falsified": falsified,
            }
            torch_kda.append(cell)
            sole_cause_falsified.append(falsified)
            no_conv_cell = index[("torch_kda_no_conv_history_cuda", kda_arm, seed, length)][
                "logits"
            ]
            no_conv_retained = (
                no_conv_cell["relative_rms_error"] / torch_cell["relative_rms_error"]
                if torch_cell["relative_rms_error"]
                else None
            )
            no_conv_falsified = bool(no_conv_cell["argmax_disagreements"]) or (
                no_conv_retained is not None and no_conv_retained >= 0.5
            )
            no_conv_history.append(
                {
                    "seed": seed,
                    "tokens": length,
                    "torch_relative_rms_error": torch_cell["relative_rms_error"],
                    "no_conv_history_relative_rms_error": no_conv_cell["relative_rms_error"],
                    "no_conv_history_over_torch_relative_rms": no_conv_retained,
                    "no_conv_history_argmax_disagreements": no_conv_cell["argmax_disagreements"],
                    "convolution_history_sole_cause_falsified": no_conv_falsified,
                }
            )
    cpu = [index[("cpu_sentinel", arm, 42, 8)]["logits"] for arm in contract["arms"]]
    return {
        "native_cuda": native,
        "torch_kda_cuda": torch_kda,
        "fla_sole_cause_falsified_in_all_cells": all(sole_cause_falsified),
        "torch_kda_no_conv_history_cuda": no_conv_history,
        "convolution_history_sole_cause_falsified_in_all_cells": all(
            cell["convolution_history_sole_cause_falsified"] for cell in no_conv_history
        ),
        "cpu_state_semantics_bug_signal": any(not cell["passed"] for cell in cpu),
        "cpu_sentinels": [{"arm_id": arm, **cell} for arm, cell in zip(contract["arms"], cpu)],
    }


def run(contract_path, runner_revision):
    contract_path, contract, matrix_path, matrix = load_contract(contract_path)
    if (
        any(configuration["device"] == "cuda" for configuration in contract["execution_matrices"])
        and not torch.cuda.is_available()
    ):
        raise RuntimeError("CUDA decode diagnostic requires CUDA")
    repository_root = contract_path.parents[2]
    output_root = repository_root / matrix["planned_primary_baselines"]["output_root"]
    arms = {arm["id"]: arm for arm in matrix["planned_primary_baselines"]["arms"]}
    numerical = contract["numerical_contract"]
    results = []
    for configuration in contract["execution_matrices"]:
        for arm_id in configuration["arms"]:
            results.append(
                run_configuration(
                    configuration,
                    arms[arm_id],
                    output_root,
                    numerical["relative_tolerance"],
                    numerical["absolute_tolerance"],
                )
            )
    summary = summarize(results, contract)
    return {
        "format": "speck_cuda_decode_diagnostic",
        "format_version": 1,
        "status": "complete_failure_classification",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runner_revision": runner_revision,
        "contract": str(contract_path.relative_to(repository_root)),
        "contract_sha256": file_sha256(contract_path),
        "baseline_matrix_sha256": file_sha256(matrix_path),
        "hardware": (
            {
                "device": torch.cuda.get_device_name(),
                "capability": list(torch.cuda.get_device_capability()),
                "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                "driver": command_output(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
                ).splitlines()[0],
            }
            if torch.cuda.is_available()
            else None
        ),
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "flash_linear_attention": importlib.metadata.version("flash-linear-attention"),
        },
        "results": results,
        "summary": summary,
        "decision_scope": contract["decision_scope"],
    }


def main(argv=None):
    args = arguments(argv)
    report = run(args.contract, repository_revision())
    atomic_json(args.output, report)
    print(f"completed {len(report['results'])} CUDA decode diagnostic configurations")


if __name__ == "__main__":
    main()
