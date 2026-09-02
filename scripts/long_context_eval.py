"""Run deterministic long-context quality, latency, and memory curves."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

from scripts.infer import load_checkpoint_model
from speck.architecture import AttentionSpec
from speck.checkpoint import checkpoint_identity, latest
from speck.common import base_dir
from speck.config import load_experiment
from speck.long_context import (
    aggregate_results,
    build_passkey_case,
    evaluate_case,
    parse_lengths,
    validate_eval_settings,
)
from speck.tokenizer import get_tokenizer


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--lengths",
        type=parse_lengths,
        default=None,
        help="comma-separated pilot lengths overriding long_context.json",
    )
    parser.add_argument(
        "--depths",
        default=None,
        help="comma-separated needle depths overriding long_context.json",
    )
    parser.add_argument(
        "--samples-per-depth",
        type=int,
        default=None,
        help="pilot sample count overriding long_context.json",
    )
    parser.add_argument(
        "--warmup-each-length",
        action="store_true",
        help="run one unmeasured case before timing each distinct prompt length",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_depths(value):
    try:
        depths = [float(item) for item in value.split(",")]
    except (AttributeError, ValueError) as error:
        raise ValueError("depths must be comma-separated numbers") from error
    if not depths:
        raise ValueError("depths must be comma-separated numbers")
    return depths


def resolved_eval_settings(config, args):
    values = dict(config)
    if args.lengths is not None:
        values["lengths"] = list(args.lengths)
    if args.depths is not None:
        values["depths"] = parse_depths(args.depths)
    if args.samples_per_depth is not None:
        values["samples_per_depth"] = args.samples_per_depth
    return validate_eval_settings(values)


def report_config(settings):
    return {
        "lengths": list(settings["lengths"]),
        "depths": list(settings["depths"]),
        "samples_per_depth": settings["samples_per_depth"],
        "effective_threshold": settings["effective_threshold"],
        "kv_cache_dtype": settings["kv_cache_dtype"],
    }


def run(args):
    configs = load_experiment(args.experiment, "long_context", "tokenizer", "train")
    settings = resolved_eval_settings(configs["long_context"], args)
    checkpoint_dir = args.checkpoint_dir or Path(
        configs["train"].get("output_dir")
        or Path(base_dir()) / "checkpoints" / configs["train"]["run"]
    )
    step = args.step if args.step is not None else latest(checkpoint_dir)
    if step is None:
        raise FileNotFoundError(f"no checkpoint found in {checkpoint_dir}")
    device = torch.device(args.device)
    model, metadata = load_checkpoint_model(checkpoint_dir, step, device)
    tokenizer = get_tokenizer(**configs["tokenizer"])
    if max(settings["lengths"]) > model.config.max_position_embeddings:
        raise ValueError("evaluation length exceeds the model's allocated context")
    attention_scopes = sorted(
        {
            branch.scope
            for invocation in model.execution_plan
            for stage in invocation.block.stages
            for branch in stage.branches
            if isinstance(branch, AttentionSpec)
        }
    )
    results = []
    kv_cache_dtype = getattr(torch, settings["kv_cache_dtype"])
    for length in settings["lengths"]:
        if args.warmup_each_length:
            warmup_case = build_passkey_case(tokenizer, length, seed=0, depth=0.5)
            evaluate_case(
                model,
                warmup_case,
                device=device,
                kv_cache_dtype=kv_cache_dtype,
            )
            print(f"{length:,} warmup complete")
        for depth_index, depth in enumerate(settings["depths"]):
            for sample in range(settings["samples_per_depth"]):
                seed = 10_000 * depth_index + sample
                case = build_passkey_case(tokenizer, length, seed, depth)
                result = evaluate_case(
                    model,
                    case,
                    device=device,
                    kv_cache_dtype=kv_cache_dtype,
                )
                results.append(result)
                print(
                    f"{length:,} depth={depth:.2f} seed={seed} "
                    f"exact={result['exact_match']:.0f} "
                    f"choice={result['candidate_accuracy']:.0f} "
                    f"rank={result['candidate_rank']} "
                    f"prefill={result['prefill_seconds']:.3f}s"
                )
    summary = aggregate_results(results, settings["effective_threshold"])
    report = {
        "format": "speck_long_context_evaluation",
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": str(args.experiment.expanduser().resolve()),
        "checkpoint": checkpoint_identity(checkpoint_dir, step),
        "config": report_config(settings),
        "model_context": model.config.max_position_embeddings,
        "device": str(device),
        "torch_version": torch.__version__,
        "warmup_each_length": args.warmup_each_length,
        "positional_regime": {
            "attention_scopes": attention_scopes,
            "rope_scaling_factor": model.config.rope_scaling_factor,
            "training_sequence_length": configs["train"]["sequence_length"],
            "extrapolates_global_rope": "global" in attention_scopes
            and max(settings["lengths"]) > configs["train"]["sequence_length"],
        },
        "results": results,
        **summary,
    }
    output = (
        args.output
        or Path(base_dir())
        / "evaluations"
        / "long-context"
        / configs["train"]["run"]
        / f"{step}.json"
    )
    atomic_json(output, report)
    return report


def main(argv=None):
    report = run(arguments(argv))
    print(f"effective context: {report['effective_length']}")


if __name__ == "__main__":
    main()
