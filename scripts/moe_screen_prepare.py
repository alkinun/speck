"""Prepare the active-parameter-matched mixture-of-experts design screen.

Every arm keeps the source architecture's mixer stack and its active feed-forward
width, so the dense control is the unmodified source model and each routed arm
spends the same per-token compute. Granularity arms additionally hold total
parameters fixed by scaling expert width inversely with the expert count.
"""

import argparse
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import torch

from speck.architecture import (
    ArchitectureConfig,
    RoutedSwiGLUSpec,
    StageConfig,
    SwiGLUSpec,
)
from speck.config import load_experiment
from speck.model import SpeckForCausalLM

CORPUS = "dclm-edu-1b-specklabs-moe-sweep"


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_experiment", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--train-tokens", type=int, default=500_000_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def dense_width(source):
    """Return the single feed-forward width every arm must keep active."""

    widths = {
        branch.intermediate_size
        for group in source.blocks
        for stage in group.block.stages
        for branch in stage.branches
        if isinstance(branch, SwiGLUSpec)
    }
    if len(widths) != 1:
        raise ValueError("the source architecture must use one feed-forward width")
    return widths.pop()


def routed_stage(width, num_experts, top_k, shared=False):
    """Build a feed-forward stage whose active width equals the dense width."""

    if width % top_k:
        raise ValueError("active width must divide evenly across the selected experts")
    expert_width = width // top_k
    if shared:
        # One always-on expert plus one routed selection, at a matched total bank.
        branches = (
            RoutedSwiGLUSpec(expert_width, num_experts - 1, 1),
            SwiGLUSpec(expert_width),
        )
    else:
        branches = (RoutedSwiGLUSpec(expert_width, num_experts, top_k),)
    return StageConfig(branches)


def apply_feed_forward(source, stage_for_block):
    """Rewrite every feed-forward stage, leaving mixer stages untouched."""

    groups = []
    for index, group in enumerate(source.blocks):
        stages = tuple(
            stage_for_block(index) if _is_feed_forward(stage) else stage
            for stage in group.block.stages
        )
        groups.append(replace(group, block=replace(group.block, stages=stages)))
    return replace(source, blocks=tuple(groups), expected_parameters=None)


def _is_feed_forward(stage):
    return all(isinstance(branch, SwiGLUSpec) for branch in stage.branches)


def arm_definitions(width, experts=8, top_k=2):
    """Return every screen arm as a name, question, and per-block stage rule."""

    dense = StageConfig((SwiGLUSpec(width),))
    routed = routed_stage(width, experts, top_k)
    granularity = tuple(
        (
            f"g{count}",
            "granularity",
            (lambda stage: lambda _index: stage)(routed_stage(width, count, selected)),
        )
        # Fine-grained segmentation: split each expert m ways and select m times as
        # many, which holds both active and total expert parameters fixed.
        for count, selected in (
            (experts, top_k),
            (experts * 2, top_k * 2),
            (experts * 4, top_k * 4),
        )
    )
    return (
        ("dense", "control", lambda _index: dense),
        *granularity,
        (
            "p-dense-first",
            "placement",
            lambda index: dense if index < 2 else routed,
        ),
        (
            "p-interleaved",
            "placement",
            lambda index: routed if index % 2 == 0 else dense,
        ),
        (
            "p-shared",
            "placement",
            lambda _index: routed_stage(width, experts, top_k, shared=True),
        ),
    )


def router_parameters(config):
    """Return the router weights, which are active in every routed arm."""

    return sum(
        invocation.block.hidden_size * branch.num_experts
        for invocation in config.execution_plan
        for stage in invocation.block.stages
        for branch in stage.branches
        if isinstance(branch, RoutedSwiGLUSpec)
    )


def accounting(config):
    with torch.device("meta"):
        model = SpeckForCausalLM(config)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "parameters": total,
        "active_parameters": config.active_parameter_count(total),
        "flops_per_token_at_2048": model.flops_per_token(2_048),
        "router_parameters": router_parameters(config),
    }


def train_settings(name, train_tokens, seed, routed):
    steps = train_tokens // 65_536
    return {
        "batch_tokens": 65_536,
        # Operational only; accumulation absorbs the difference and recomputation is
        # mathematically transparent. Routed arms hold three times the weights and
        # replicate every token top-k ways through the permutation, so they take the
        # smaller batch. Checkpointing is uniform so no arm is memory-advantaged.
        "activation_checkpointing": True,
        "device_batch_size": 4 if routed else 8,
        "diagnostics_every": 100,
        "eval_every": max(steps // 10, 1),
        "eval_tokens": 20_000_000,
        "final_eval_tokens": 20_000_000,
        "grad_clip": 1.0,
        "load_balance_coefficient": 0.01,
        "log_every": 10,
        "lr": 0.0015,
        "lr_schedule": "cosine",
        "min_lr": 0.05,
        "optimizer": "muon",
        "output_dir": None,
        "router_z_loss_coefficient": 0.001,
        "run": f"SpeckLC-150M-MoEScreen-{name}",
        "save_every": 0,
        "seed": seed,
        "sequence_length": 2_048,
        "train_tokens": train_tokens,
        "wandb_group": "SpeckLC-150M-MoEScreen",
        "wandb_project": "speck-moe-screen",
        "warmup_steps": 102,
        "weight_decay": 0.1,
    }


def prepare(args):
    output = args.output_dir
    if output.exists():
        raise FileExistsError(f"screen already exists: {output}")
    source_settings = load_experiment(args.source_experiment, "model")["model"]
    source = ArchitectureConfig.from_dict(source_settings)
    tokenizer = load_experiment(args.source_experiment, "tokenizer")["tokenizer"]
    data = json.loads((Path(__file__).parent / "moe_screen_data.json").read_text())
    if data["output_name"] != CORPUS:
        raise ValueError("the screen corpus must be the prepared DCLM-Edu stream")

    width = dense_width(source)
    arms = {}
    for name, question, stage_for_block in arm_definitions(width):
        config = apply_feed_forward(source, stage_for_block)
        arms[name] = {
            "question": question,
            "model": config.export(),
            **accounting(config),
        }

    # Routers are genuinely active, so the matched quantity is active parameters
    # excluding routers. That is what fixes per-token feed-forward compute. A shared
    # expert adds one normalization per block, which is allowed but recorded.
    control = arms["dense"]["active_parameters"]
    for arm in arms.values():
        arm["matched_active_parameters"] = arm["active_parameters"] - arm["router_parameters"]
        arm["matched_active_delta"] = arm["matched_active_parameters"] - control
    mismatched = [
        name for name, arm in arms.items() if abs(arm["matched_active_delta"]) > control // 2_000
    ]
    if mismatched:
        raise ValueError(f"active parameters are not matched: {', '.join(sorted(mismatched))}")

    contract = {
        "format": "speck_moe_design_screen",
        "format_version": 1,
        "source_experiment": str(args.source_experiment),
        "corpus": CORPUS,
        "dense_feed_forward_width": width,
        "seed": args.seed,
        "train_tokens": args.train_tokens,
        "arms": {
            name: {key: value for key, value in arm.items() if key != "model"}
            for name, arm in arms.items()
        },
    }

    building = output.with_name(output.name + ".building")
    shutil.rmtree(building, ignore_errors=True)
    try:
        building.mkdir(parents=True)
        for name, arm in arms.items():
            directory = building / name
            directory.mkdir()
            files = {
                "model.json": arm["model"],
                "data.json": data,
                "tokenizer.json": tokenizer,
                "train.json": train_settings(
                    name, args.train_tokens, args.seed, arm["router_parameters"] > 0
                ),
            }
            for filename, values in files.items():
                (directory / filename).write_text(
                    json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
        (building / "screen.json").write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(building, output)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return contract


def main(argv=None):
    contract = prepare(arguments(argv))
    width = max(len(name) for name in contract["arms"])
    header = f"{'arm'.ljust(width)}  {'total':>13} {'active':>13} {'matched':>13} {'delta':>7}"
    print(f"{header}  question")
    for name, arm in contract["arms"].items():
        print(
            f"{name.ljust(width)}  {arm['parameters']:13,} {arm['active_parameters']:13,} "
            f"{arm['matched_active_parameters']:13,} {arm['matched_active_delta']:7,}  "
            f"{arm['question']}"
        )


if __name__ == "__main__":
    main()
