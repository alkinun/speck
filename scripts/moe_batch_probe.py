"""Qualify one compiled synthetic training microbatch against a VRAM ceiling."""

import argparse
import json
import time
from pathlib import Path

import torch

from speck.config import load_experiment
from speck.model import build_model


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--vram-fraction", type=float, default=0.9)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--max-autotune", action="store_true")
    parser.add_argument("--compile-optimizer", action="store_true")
    parser.add_argument(
        "--production-optimizer",
        action="store_true",
        help="compile Muon with the production Inductor settings instead of AOT eager",
    )
    args = parser.parse_args(argv)
    if args.batch_size not in (16, 8, 4, 2, 1):
        parser.error("--batch-size must be one of 16, 8, 4, 2, 1")
    if not 0 < args.vram_fraction <= 1:
        parser.error("--vram-fraction must be in (0, 1]")
    return args


def probe(
    experiment,
    batch_size,
    vram_fraction=0.9,
    compile_model=True,
    max_autotune=False,
    compile_optimizer=False,
    production_optimizer=False,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for device-batch qualification")
    configs = load_experiment(experiment, "model", "train")
    training = configs["train"]
    torch.manual_seed(training["seed"])
    device = torch.device("cuda")
    model = build_model(configs["model"], vocab_size=32_000).to(device)
    model.init_weights()
    optimizer = model.optimizer(
        training["lr"], training["weight_decay"], training["optimizer"]
    )
    if compile_model:
        options = (
            {
                "max_autotune": True,
                "coordinate_descent_tuning": True,
                "aggressive_fusion": True,
            }
            if max_autotune
            else {}
        )
        model = torch.compile(
            model,
            dynamic=False,
            options=options,
        )
        compile_step = getattr(optimizer, "compile_step", None)
        if compile_optimizer and compile_step is not None:
            if production_optimizer:
                compile_step()
            else:
                # PyTorch 2.11 has an Inductor scheduler regression for the banked
                # Muon graph; AOT eager still qualifies graph capture/execution.
                compile_step(backend="aot_eager")
    tokens = torch.randint(
        0,
        32_000,
        (batch_size, training["sequence_length"]),
        device=device,
    )
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    output = model(
        tokens,
        tokens,
        return_training_output=True,
        load_balance_coefficient=training["load_balance_coefficient"],
        router_z_loss_coefficient=training["router_z_loss_coefficient"],
    )
    output.total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip"])
    optimizer.step()
    torch.cuda.synchronize(device)
    duration = time.perf_counter() - started
    allocated = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.max_memory_reserved(device)
    total = torch.cuda.get_device_properties(device).total_memory
    return {
        "architecture": experiment.name,
        "batch_size": batch_size,
        "compiled": compile_model,
        "compiled_optimizer": compile_optimizer,
        "optimizer_backend": (
            "inductor" if production_optimizer else "aot_eager"
        )
        if compile_optimizer
        else None,
        "duration_seconds": duration,
        "loss": output.total_loss.item(),
        "peak_allocated_mib": allocated / 2**20,
        "peak_reserved_mib": reserved / 2**20,
        "total_vram_mib": total / 2**20,
        "vram_fraction_limit": vram_fraction,
        "qualified": torch.isfinite(output.total_loss).item()
        and reserved <= vram_fraction * total,
    }


def main(argv=None):
    args = arguments(argv)
    result = probe(
        args.experiment.expanduser().resolve(),
        args.batch_size,
        args.vram_fraction,
        not args.no_compile,
        args.max_autotune,
        args.compile_optimizer,
        args.production_optimizer,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["qualified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
