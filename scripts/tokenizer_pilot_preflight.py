"""Measure exact 60M tokenizer-pilot training shapes before launching quality runs."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from speck.io import atomic_json, file_sha256
from speck.model import build_model
from speck.scale_targets import flop_accounting, model_settings
from speck.train import optimization_step


def _bound(identity, context):
    path = Path(identity["path"]).resolve()
    if not path.is_file() or file_sha256(path) != identity["sha256"]:
        raise ValueError(f"{context} identity mismatch")
    return path


class _SyntheticLoader:
    def __init__(self, vocab_size, sequence_length, device, seed):
        generator = torch.Generator(device=device).manual_seed(seed)
        self.batches = []
        for index in range(4):
            tokens = torch.randint(
                0,
                vocab_size,
                (1, sequence_length + 1),
                generator=generator,
                device=device,
            )
            self.batches.append(
                (
                    tokens[:, :-1].contiguous(),
                    tokens[:, 1:].contiguous(),
                    {"selected_source": "synthetic", "index": index},
                )
            )
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        value = self.batches[self.index % len(self.batches)]
        self.index += 1
        return value


def _fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def run_preflight(plan):
    if plan.get("format") != "speck_tokenizer_pilot_preflight" or plan.get("format_version") != 1:
        raise ValueError("unsupported tokenizer pilot preflight")
    _bound(plan["pilot_plan"], "pilot plan")
    scale_path = _bound(plan["scale_spec"], "scale spec")
    settings = plan["settings"]
    if settings["device"] != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("tokenizer pilot preflight requires CUDA")
    if settings["batch_tokens"] != (
        settings["device_batch_size"] * settings["sequence_length"] * settings["accumulation"]
    ):
        raise ValueError("preflight batch geometry is inconsistent")
    scale = json.loads(scale_path.read_text())
    target = next(item for item in scale["targets"] if item["id"] == plan["target_id"])
    device = torch.device("cuda")
    results = []
    for declaration in plan["tokenizers"]:
        torch.manual_seed(settings["seed"])
        torch.cuda.manual_seed_all(settings["seed"])
        model = build_model(
            model_settings(target, declaration["vocab_size"]),
            declaration["vocab_size"],
            declaration["bos_token_id"],
            declaration["eos_token_id"],
            loss_backend=settings["loss_backend"],
        ).to(device=device, dtype=torch.bfloat16)
        model.init_weights()
        parameters = tuple(model.parameters())
        optimizer = model.optimizer(
            settings["learning_rate"], settings["weight_decay"], settings["optimizer"]
        )
        train_model = (
            torch.compile(
                model,
                dynamic=False,
                options={
                    "max_autotune": True,
                    "coordinate_descent_tuning": True,
                    "aggressive_fusion": True,
                },
            )
            if settings["compile"]
            else model
        )
        compile_step = getattr(optimizer, "compile_step", None)
        if settings["compile"] and compile_step is not None:
            compile_step()
        loader = iter(
            _SyntheticLoader(
                declaration["vocab_size"],
                settings["sequence_length"],
                device,
                settings["seed"],
            )
        )
        batch = next(loader)
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(settings["warmup_optimizer_steps"]):
            _, _, batch = optimization_step(
                train_model,
                parameters,
                optimizer,
                loader,
                batch,
                settings["accumulation"],
                settings["grad_clip"],
                settings["learning_rate"],
            )
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        losses = []
        for _ in range(settings["measured_optimizer_steps"]):
            loss, _, batch = optimization_step(
                train_model,
                parameters,
                optimizer,
                loader,
                batch,
                settings["accumulation"],
                settings["grad_clip"],
                settings["learning_rate"],
            )
            losses.append(float(loss))
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        tokens = settings["batch_tokens"] * settings["measured_optimizer_steps"]
        throughput = tokens / elapsed
        analytic = flop_accounting(
            target,
            declaration["vocab_size"],
            settings["sequence_length"],
            contract_version=2,
        )["analytic_training_flops_per_token"]
        results.append(
            {
                **declaration,
                "parameters": sum(parameter.numel() for parameter in parameters),
                "analytic_training_flops_per_token": analytic,
                "measured_seconds": elapsed,
                "measured_tokens": tokens,
                "tokens_per_second": throughput,
                "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
                "losses": losses,
                "projected_fixed_document_hours": declaration["fixed_document_aligned_tokens"]
                / throughput
                / 3600,
                "projected_run_hours": declaration["run_stop_aligned_tokens"] / throughput / 3600,
            }
        )
        del train_model, optimizer, model, parameters, loader, batch
        torch.cuda.empty_cache()
    by_id = {item["id"]: item for item in results}
    baseline_hours = by_id[plan["baseline_id"]]["projected_run_hours"]
    custom_ids = [item["id"] for item in plan["tokenizers"] if item["id"] != plan["baseline_id"]]
    screen_hours = baseline_hours + sum(by_id[item]["projected_run_hours"] for item in custom_ids)
    worst_matrix_hours = (
        screen_hours
        + 2 * baseline_hours
        + 2 * max(by_id[item]["projected_run_hours"] for item in custom_ids)
    )
    result = {
        "format": "speck_tokenizer_pilot_preflight_result",
        "format_version": 1,
        "status": (
            "throughput_projection_inside_30_hour_ceiling"
            if worst_matrix_hours <= plan["gpu_hour_ceiling"]
            else "throughput_projection_exceeds_30_hour_ceiling"
        ),
        "plan_fingerprint": _fingerprint(plan),
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
        "tokenizers": results,
        "screen_projected_gpu_hours": screen_hours,
        "worst_case_seven_run_projected_gpu_hours": worst_matrix_hours,
        "gpu_hour_ceiling": plan["gpu_hour_ceiling"],
        "projection_authority": "short synthetic steady-state estimate only; real data and evaluation overhead remain unmeasured",
        "quality_run_authority": worst_matrix_hours <= plan["gpu_hour_ceiling"],
        "D5_tokenizer_audit": "unopened",
    }
    atomic_json(plan["output"], result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan")
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    result = run_preflight(plan)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
