"""Prepare a 4K language-model staircase that isolates Kimi Linear design choices."""

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
    GatedDeltaNetSpec,
    KimiDeltaAttentionSpec,
    StageConfig,
)
from speck.config import load_experiment
from speck.model import SpeckForCausalLM

VARIANTS = (
    "gdn-fla-silu-rope",
    "gdn-fla-sigmoid-rope",
    "gdn-fla-sigmoid-nope",
    "kda-sigmoid-nope",
)
INTERVENTIONS = {
    "gdn-fla-silu-rope": "replace historical Speck decay initialization with FLA timescales",
    "gdn-fla-sigmoid-rope": "replace the SiLU output gate with sigmoid",
    "gdn-fla-sigmoid-nope": "remove RoPE from all five global attention layers",
    "kda-sigmoid-nope": "replace scalar GDN decay with channel-wise KDA decay",
}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_experiment", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args(argv)


def variant_architecture(source, variant):
    if variant not in VARIANTS:
        raise ValueError(f"unknown Kimi transfer variant: {variant}")
    use_sigmoid = variant != "gdn-fla-silu-rope"
    use_nope = variant in {"gdn-fla-sigmoid-nope", "kda-sigmoid-nope"}
    use_kda = variant == "kda-sigmoid-nope"
    groups = []
    for invocation in source.execution_plan:
        stages = []
        for stage in invocation.block.stages:
            branches = []
            for branch in stage.branches:
                if isinstance(branch, GatedDeltaNetSpec):
                    if use_kda:
                        branch = KimiDeltaAttentionSpec(
                            key_head_dim=branch.key_head_dim,
                            value_head_dim=branch.value_head_dim,
                            num_key_heads=branch.num_key_heads,
                            num_value_heads=branch.num_value_heads,
                            conv_kernel_size=branch.conv_kernel_size,
                        )
                    else:
                        branch = replace(
                            branch,
                            decay_initialization="fla",
                            output_gate_activation="sigmoid" if use_sigmoid else "silu",
                        )
                elif isinstance(branch, AttentionSpec) and use_nope:
                    branch = replace(branch, rope_dim=0)
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
    return replace(config, expected_parameters=parameters)


def staircase_summary(source, variants, sequence_length, train_tokens):
    with torch.device("meta"):
        source_model = SpeckForCausalLM(source)
    target_flops = source_model.flops_per_token(sequence_length) * train_tokens
    result = {}
    for name, config in variants.items():
        with torch.device("meta"):
            model = SpeckForCausalLM(config)
        flops = model.flops_per_token(sequence_length)
        result[name] = {
            "parameters": model.parameter_count(),
            "flops_per_token": flops,
            "compute_matched_tokens": target_flops // flops,
        }
    return result


def prepare(args):
    source_dir = args.source_experiment.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Kimi transfer family already exists: {output}")
    configs = load_experiment(
        source_dir,
        "data",
        "long_context",
        "model",
        "tokenizer",
        "train",
    )
    source = ArchitectureConfig.from_dict(configs["model"])
    variants = {name: variant_architecture(source, name) for name in VARIANTS}
    train_tokens = configs["train"]["train_tokens"]
    sequence_length = configs["train"]["sequence_length"]
    summary = staircase_summary(source, variants, sequence_length, train_tokens)
    contract = {
        "format": "speck_kimi_transfer_staircase",
        "format_version": 1,
        "comparison": "token-matched one-intervention staircase with compute-matched budgets",
        "source_experiment": str(source_dir),
        "existing_control": {
            "experiment": str(source_dir),
            "variant": "gdn-speck-silu-rope",
        },
        "sequence_length": sequence_length,
        "train_tokens": train_tokens,
        "seed": 42,
        "intervention_order": [
            {"variant": name, "change_from_previous": INTERVENTIONS[name]}
            for name in VARIANTS
        ],
        "variants": summary,
        "result": "../../results/SpeckLC-150M-KimiTransfer131M/summary.json",
        "status": "prepared",
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
                "seed": 42,
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
        (building / "staircase.json").write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(building, output)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return contract


def main(argv=None):
    args = arguments(argv)
    contract = prepare(args)
    print(f"Prepared {len(contract['variants'])} Kimi transfer variants under {args.output_dir}")


if __name__ == "__main__":
    main()
