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
from speck.dataset import validate_data_settings
from speck.model import SpeckForCausalLM

VARIANTS = (
    "gdn-global",
    "gdn-local",
    "pure-gdn",
    "conv-global",
    "full-local",
    "full-global",
)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_experiment", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--window-size", type=int, default=4_096)
    parser.add_argument(
        "--train-tokens",
        type=int,
        default=None,
        help="override the source training horizon for a screening rung",
    )
    parser.add_argument(
        "--data-tokens",
        type=int,
        default=None,
        help="scale source mixture phase boundaries for a smaller shared pilot corpus",
    )
    parser.add_argument(
        "--data-experiment",
        type=Path,
        default=None,
        help="reuse the packed-data contract from another experiment",
    )
    return parser.parse_args(argv)


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def scaled_integer(value, source_tokens, target_tokens, *, minimum=1):
    return max(minimum, round(value * target_tokens / source_tokens))


def scale_data_config(data, requested_tokens):
    """Scale phase durations exactly while preserving every source weight."""

    requested_tokens = positive_integer(requested_tokens, "pilot data tokens")
    source_tokens = data["requested_train_tokens"]
    phases = []
    for phase in data["mixture"]["phases"]:
        numerator = phase["end_tokens"] * requested_tokens
        if numerator % source_tokens:
            raise ValueError("pilot data tokens do not scale source phase boundaries to integers")
        phases.append(
            {
                **phase,
                "end_tokens": numerator // source_tokens,
            }
        )
    output_name = data.get("output_name") or "packed"
    scaled = {
        **data,
        "mixture": {**data["mixture"], "phases": phases},
        "output_dir": None,
        "output_name": f"{output_name}-Pilot-{requested_tokens}",
        "requested_train_tokens": requested_tokens,
    }
    validate_data_settings(
        sources=scaled["sources"],
        mixture=scaled["mixture"],
        requested_train_tokens=scaled["requested_train_tokens"],
        validation_tokens_per_source=scaled["validation_tokens_per_source"],
        validation_fraction=scaled["validation_fraction"],
        filtering=scaled["filtering"],
        dedup=scaled["dedup"],
        shards=scaled["shards"],
    )
    return scaled


def scale_train_config(train, train_tokens):
    train_tokens = positive_integer(train_tokens, "pilot train tokens")
    source_tokens = train["train_tokens"]
    return {
        **train,
        "train_tokens": train_tokens,
        "warmup_steps": scaled_integer(
            train["warmup_steps"], source_tokens, train_tokens, minimum=0
        ),
    }


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
    gdn_template = next(
        branch
        for invocation in source.execution_plan
        for stage in invocation.block.stages
        for branch in stage.branches
        if isinstance(branch, GatedDeltaNetSpec)
    )
    groups = []
    for invocation in source.execution_plan:
        stages = list(invocation.block.stages)
        first = stages[0]
        if len(first.branches) != 1:
            raise ValueError("mixer ablations require one mixer in the first stage")
        mixer = first.branches[0]
        if variant in {"full-global", "full-local"}:
            local = variant == "full-local"
            mixer = replace(
                attention_template,
                scope="sliding" if local else "global",
                window_size=window_size if local else None,
            )
        elif variant == "pure-gdn":
            mixer = gdn_template
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
    data_experiment_arg = getattr(args, "data_experiment", None)
    data_experiment = (
        source_dir if data_experiment_arg is None else data_experiment_arg.expanduser().resolve()
    )
    if data_experiment == source_dir:
        data_configs = {"data": configs["data"], "tokenizer": configs["tokenizer"]}
    else:
        data_configs = load_experiment(data_experiment, "data", "tokenizer")
    if data_configs["tokenizer"] != configs["tokenizer"]:
        raise ValueError("mixer sweep data tokenizer does not match the source experiment")
    train_config = (
        configs["train"]
        if args.train_tokens is None
        else scale_train_config(configs["train"], args.train_tokens)
    )
    data_config = (
        data_configs["data"]
        if args.data_tokens is None
        else scale_data_config(data_configs["data"], args.data_tokens)
    )
    if train_config["train_tokens"] > data_config["requested_train_tokens"]:
        raise ValueError("mixer sweep training horizon exceeds the prepared data horizon")
    source = ArchitectureConfig.from_dict(configs["model"])
    variants = {name: variant_architecture(source, name, args.window_size) for name in VARIANTS}
    summary = ablation_summary(
        source,
        variants,
        train_config["sequence_length"],
        train_config["train_tokens"],
    )
    contract = {
        "format": "speck_mixer_ablation",
        "format_version": 1,
        "source_experiment": str(source_dir),
        "data_experiment": str(data_experiment),
        "window_size": args.window_size,
        "comparison": "token-matched configs with separately reported compute-matched budgets",
        "train_tokens": train_config["train_tokens"],
        "data_tokens": data_config["requested_train_tokens"],
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
                **train_config,
                "output_dir": None,
                "run": f"{output.name}-{name}",
                "wandb_group": output.name,
            }
            materialized = {
                "data.json": data_config,
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
