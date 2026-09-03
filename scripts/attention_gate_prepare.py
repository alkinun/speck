"""Prepare a parameter- and FLOP-matched global-attention gate screen."""

import argparse
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import torch

from speck.architecture import (
    ArchitectureConfig,
    AttentionSpec,
    BlockConfig,
    BlockGroup,
    StageConfig,
    SwiGLUSpec,
)
from speck.config import load_experiment
from speck.model import SpeckForCausalLM

VARIANTS = ("ungated", "headwise", "elementwise")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_experiment", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--train-tokens", type=int, default=32_000_000)
    return parser.parse_args(argv)


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def gated_architecture(source, gate):
    if gate not in VARIANTS:
        raise ValueError(f"unknown attention gate variant: {gate}")
    attention_parameters = 0
    ffn_parameter_step = 0
    for invocation in source.execution_plan:
        hidden_size = invocation.block.hidden_size
        for stage in invocation.block.stages:
            for branch in stage.branches:
                if isinstance(branch, AttentionSpec):
                    if gate == "headwise":
                        attention_parameters += hidden_size * (hidden_size // branch.head_dim)
                    elif gate == "elementwise":
                        attention_parameters += hidden_size * hidden_size
                elif isinstance(branch, SwiGLUSpec):
                    ffn_parameter_step += 3 * hidden_size
    if attention_parameters % ffn_parameter_step:
        raise ValueError("attention gate parameters cannot be matched by a uniform FFN decrement")
    ffn_decrement = attention_parameters // ffn_parameter_step
    groups = []
    for invocation in source.execution_plan:
        stages = []
        for stage in invocation.block.stages:
            branches = []
            for branch in stage.branches:
                if isinstance(branch, AttentionSpec):
                    branch = replace(branch, output_gate=gate if gate != "ungated" else "none")
                elif isinstance(branch, SwiGLUSpec):
                    branch = replace(
                        branch, intermediate_size=branch.intermediate_size - ffn_decrement
                    )
                    if branch.intermediate_size < 1:
                        raise ValueError("attention gate parameter matching removed the FFN")
                branches.append(branch)
            stages.append(StageConfig(tuple(branches)))
        groups.append(
            BlockGroup(
                BlockConfig(invocation.block.hidden_size, tuple(stages)),
                repeat=1,
                weight_sharing="none",
            )
        )
    config = replace(source, blocks=tuple(groups), expected_parameters=None)
    with torch.device("meta"):
        parameters = SpeckForCausalLM(config).parameter_count()
    return replace(config, expected_parameters=parameters), ffn_decrement


def prepare(args):
    source_dir = args.source_experiment.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    train_tokens = positive_integer(args.train_tokens, "training tokens")
    if output.exists():
        raise FileExistsError(f"attention gate screen already exists: {output}")
    configs = load_experiment(source_dir, "data", "long_context", "model", "tokenizer", "train")
    source = ArchitectureConfig.from_dict(configs["model"])
    variants = {name: gated_architecture(source, name) for name in VARIANTS}
    sequence_length = configs["train"]["sequence_length"]
    with torch.device("meta"):
        source_model = SpeckForCausalLM(source)
    source_parameters = source_model.parameter_count()
    source_flops = source_model.flops_per_token(sequence_length)
    summary = {}
    for name, (config, ffn_decrement) in variants.items():
        with torch.device("meta"):
            model = SpeckForCausalLM(config)
        if model.parameter_count() != source_parameters:
            raise ValueError("attention gate variant is not parameter matched")
        if model.flops_per_token(sequence_length) != source_flops:
            raise ValueError("attention gate variant is not FLOP matched")
        summary[name] = {
            "parameters": model.parameter_count(),
            "flops_per_token": model.flops_per_token(sequence_length),
            "ffn_intermediate_decrement": ffn_decrement,
        }
    contract = {
        "format": "speck_attention_gate_screen",
        "format_version": 1,
        "comparison": "parameter- and analytic-FLOP-matched global attention output gates",
        "source_experiment": str(source_dir),
        "sequence_length": sequence_length,
        "train_tokens": train_tokens,
        "seed": configs["train"].get("seed", 42),
        "variants": summary,
        "result": "../../results/SpeckLC-150M-AttentionGate32M/summary.json",
        "status": "prepared",
    }
    building = output.with_name(output.name + ".building")
    shutil.rmtree(building, ignore_errors=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    building.mkdir()
    try:
        for name, (config, _) in variants.items():
            directory = building / name
            directory.mkdir()
            train = {
                **configs["train"],
                "eval_every": 122,
                "output_dir": None,
                "run": f"{output.name}-{name}",
                "train_tokens": train_tokens,
                "wandb_group": output.name,
                "warmup_steps": 3,
            }
            materialized = {
                "data.json": configs["data"],
                "long_context.json": configs["long_context"],
                "model.json": config.export(),
                "tokenizer.json": configs["tokenizer"],
                "train.json": train,
            }
            for filename, values in materialized.items():
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
    args = arguments(argv)
    contract = prepare(args)
    print(f"Prepared {len(contract['variants'])} attention gate variants under {args.output_dir}")


if __name__ == "__main__":
    main()
