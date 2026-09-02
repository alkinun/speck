"""Materialize a narrow, compute-accounted long-context mixer ablation family."""

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
    GatedCausalConvSpec,
    GatedDeltaNetSpec,
    StageConfig,
)
from speck.config import load_experiment
from speck.model import SpeckForCausalLM

VARIANTS = ("gdn-global", "gdn-local", "conv-global", "full-global")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_experiment", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--window-size", type=int, default=4_096)
    return parser.parse_args(argv)


def variant_architecture(source, variant, window_size=4_096):
    if variant not in VARIANTS:
        raise ValueError(f"unknown mixer ablation variant: {variant}")
    if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 1:
        raise ValueError("mixer ablation window size must be a positive integer")
    attention_template = next(
        branch
        for invocation in source.execution_plan
        for stage in invocation.block.stages
        for branch in stage.branches
        if isinstance(branch, AttentionSpec)
    )
    groups = []
    for invocation in source.execution_plan:
        stages = list(invocation.block.stages)
        first = stages[0]
        if len(first.branches) != 1:
            raise ValueError("mixer ablations require one mixer in the first stage")
        mixer = first.branches[0]
        if variant == "full-global":
            mixer = replace(attention_template, scope="global", window_size=None)
        elif isinstance(mixer, AttentionSpec):
            if variant == "gdn-local":
                mixer = replace(mixer, scope="sliding", window_size=window_size)
            else:
                mixer = replace(mixer, scope="global", window_size=None)
        elif isinstance(mixer, GatedDeltaNetSpec) and variant == "conv-global":
            mixer = GatedCausalConvSpec(
                mixer.num_value_heads * mixer.value_head_dim,
                mixer.conv_kernel_size,
            )
        stages[0] = StageConfig((mixer,))
        groups.append(BlockGroup(BlockConfig(invocation.block.hidden_size, tuple(stages))))
    config = replace(source, blocks=tuple(groups), expected_parameters=None)
    with torch.device("meta"):
        model = SpeckForCausalLM(config)
    return replace(config, expected_parameters=model.parameter_count())


def ablation_summary(source, variants, sequence_length, train_tokens):
    with torch.device("meta"):
        source_model = SpeckForCausalLM(source)
    target_flops = source_model.flops_per_token(sequence_length) * train_tokens
    summary = {}
    for name, config in variants.items():
        with torch.device("meta"):
            model = SpeckForCausalLM(config)
        flops = model.flops_per_token(sequence_length)
        summary[name] = {
            "parameters": model.parameter_count(),
            "flops_per_token": flops,
            "compute_matched_tokens": target_flops // flops,
        }
    return summary


def prepare(args):
    source_dir = args.source_experiment.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"mixer ablation family already exists: {output}")
    configs = load_experiment(source_dir, "data", "long_context", "model", "tokenizer", "train")
    source = ArchitectureConfig.from_dict(configs["model"])
    variants = {name: variant_architecture(source, name, args.window_size) for name in VARIANTS}
    summary = ablation_summary(
        source,
        variants,
        configs["train"]["sequence_length"],
        configs["train"]["train_tokens"],
    )
    contract = {
        "format": "speck_mixer_ablation",
        "format_version": 1,
        "source_experiment": str(source_dir),
        "window_size": args.window_size,
        "comparison": "token-matched configs with separately reported compute-matched budgets",
        "variants": summary,
    }
    building = output.with_name(output.name + ".building")
    shutil.rmtree(building, ignore_errors=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    building.mkdir()
    try:
        for name, config in variants.items():
            directory = building / name
            directory.mkdir()
            train = {
                **configs["train"],
                "output_dir": None,
                "run": f"{output.name}-{name}",
                "wandb_group": output.name,
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
                    json.dumps(values, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        (building / "sweep.json").write_text(
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
    print(f"Prepared {len(contract['variants'])} mixer variants under {args.output_dir}")


if __name__ == "__main__":
    main()
