"""Prepare a parameter- and FLOP-matched Speck Reader Attention cache staircase.

Every arm keeps the same depth, the same number of global attention operations, and the
same attention placements. Only the number of distinct key-value caches changes: a cache
is written once by a writer layer and read by the reader layers that follow it.
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
    AttentionSpec,
    BlockConfig,
    BlockGroup,
    StageConfig,
    SwiGLUSpec,
)
from speck.config import load_experiment
from speck.model import SpeckForCausalLM

STATE_LENGTHS = (4_096, 32_768, 131_072)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_experiment", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--caches", type=int, nargs="+", default=(5, 2, 1))
    parser.add_argument("--mqa-caches", type=int, nargs="*", default=())
    parser.add_argument("--far-caches", type=int, nargs="*", default=())
    parser.add_argument("--train-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def attention_slots(config):
    """Return every global attention position in execution order."""

    slots = []
    for invocation in config.execution_plan:
        for stage_index, stage in enumerate(invocation.block.stages):
            for branch_index, branch in enumerate(stage.branches):
                if not isinstance(branch, AttentionSpec):
                    continue
                if branch.scope != "global":
                    raise ValueError("shared attention memory requires global attention slots")
                slots.append((invocation.occurrence_index, stage_index, branch_index))
    return slots


def memory_plan(slot_count, caches, binding="nearest"):
    """Assign each attention slot to a writer or to a preceding writer's memory.

    `nearest` binds a reader to the closest preceding writer, which minimises the depth
    distance a memory must survive. `farthest` binds every reader to the first writer, which
    holds the number of caches and readers fixed while maximising that distance.
    """

    caches = positive_integer(caches, "cache count")
    if caches > slot_count:
        raise ValueError("cache count cannot exceed the number of attention slots")
    if binding not in {"nearest", "farthest"}:
        raise ValueError("reader binding must be nearest or farthest")
    writers = {index * slot_count // caches for index in range(caches)}
    plan = []
    written = 0
    label = None
    for slot in range(slot_count):
        if slot in writers:
            label = f"global_{written}"
            written += 1
            plan.append(("write", label))
        else:
            plan.append(("read", "global_0" if binding == "farthest" else label))
    return plan


def shared_memory_architecture(source, caches, writer_key_value_heads=None, binding="nearest"):
    """Rewrite one architecture to share `caches` key-value caches across its attention slots."""

    slots = attention_slots(source)
    if not slots:
        raise ValueError("the source architecture has no global attention slots")
    writer_slots = {index * len(slots) // caches for index in range(caches)}
    plan = dict(zip(slots, memory_plan(len(slots), caches, binding)))
    reuses_source = caches == len(slots) and writer_key_value_heads is None
    groups = []
    reclaimed_total = 0
    residual_total = 0
    readers = 0
    for invocation in source.execution_plan:
        reclaimed = 0
        rewritten = []
        for stage_index, stage in enumerate(invocation.block.stages):
            branches = []
            for branch_index, branch in enumerate(stage.branches):
                if isinstance(branch, AttentionSpec) and not reuses_source:
                    role, label = plan[(invocation.occurrence_index, stage_index, branch_index)]
                    kv_heads = writer_key_value_heads or branch.num_key_value_heads
                    reclaimed += (
                        2
                        * invocation.block.hidden_size
                        * branch.head_dim
                        * (
                            branch.num_key_value_heads
                            if role == "read"
                            else branch.num_key_value_heads - kv_heads
                        )
                    )
                    readers += int(role == "read")
                    branch = replace(
                        branch,
                        memory=label,
                        memory_role=role,
                        num_key_value_heads=kv_heads,
                    )
                branches.append(branch)
            rewritten.append(branches)
        step = (
            3
            * invocation.block.hidden_size
            * sum(
                1 for branches in rewritten for branch in branches if isinstance(branch, SwiGLUSpec)
            )
        )
        increment = reclaimed // step if step else 0
        residual_total += reclaimed - increment * step
        reclaimed_total += reclaimed
        stages = []
        for branches in rewritten:
            restored = []
            for branch in branches:
                if isinstance(branch, SwiGLUSpec) and increment:
                    branch = replace(branch, intermediate_size=branch.intermediate_size + increment)
                restored.append(branch)
            stages.append(StageConfig(tuple(restored)))
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
    summary = {
        "caches": caches,
        "readers": readers,
        "reclaimed_key_value_parameters": reclaimed_total,
        "unmatched_parameters_in_feed_forward": residual_total,
        "reuses_source_architecture": reuses_source,
        "writer_key_value_heads": writer_key_value_heads,
        "reader_binding": binding,
        "max_reader_writer_slot_distance": max(
            (
                slot_index - max(w for w in sorted(writer_slots) if w < slot_index)
                if binding == "nearest"
                else slot_index - min(writer_slots)
            )
            for slot_index, (role, _) in enumerate(memory_plan(len(slots), caches, binding))
            if role == "read"
        )
        if caches < len(slots)
        else 0,
        "max_readers_per_memory": max(
            sum(
                1
                for role, lab in memory_plan(len(slots), caches, binding)
                if role == "read" and lab == label
            )
            for label in {
                lab for role, lab in memory_plan(len(slots), caches, binding) if role == "write"
            }
        )
        if caches < len(slots)
        else 0,
    }
    return replace(config, expected_parameters=parameters), summary


def state_bytes(config):
    with torch.device("meta"):
        model = SpeckForCausalLM(config)
        return {
            str(length): model.state(
                length=length,
                device="meta",
                dtype=torch.bfloat16,
                kv_cache_dtype=torch.bfloat16,
            ).memory_report()
            for length in STATE_LENGTHS
            if length <= config.max_position_embeddings
        }


def arm_definitions(caches, mqa_caches, far_caches=()):
    arms = [(f"caches-{count}", count, None, "nearest") for count in caches]
    arms += [(f"caches-{count}-mqa1", count, 1, "nearest") for count in mqa_caches]
    arms += [(f"caches-{count}-far", count, None, "farthest") for count in far_caches]
    names = [name for name, _, _, _ in arms]
    if len(set(names)) != len(names):
        raise ValueError("duplicate reader attention arm names")
    return arms


def prepare(args):
    source_dir = args.source_experiment.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"reader attention staircase already exists: {output}")
    configs = load_experiment(source_dir, "data", "long_context", "model", "tokenizer", "train")
    source = ArchitectureConfig.from_dict(configs["model"])
    sequence_length = configs["train"]["sequence_length"]
    train_tokens = positive_integer(
        args.train_tokens if args.train_tokens is not None else configs["train"]["train_tokens"],
        "training tokens",
    )
    seed = configs["train"].get("seed", 42) if args.seed is None else args.seed
    with torch.device("meta"):
        baseline = SpeckForCausalLM(source)
    baseline_parameters = baseline.parameter_count()
    baseline_flops = baseline.flops_per_token(sequence_length)
    baseline_state = state_bytes(source)
    variants = {}
    summary = {}
    for name, caches, kv_heads, binding in arm_definitions(
        args.caches, args.mqa_caches, getattr(args, "far_caches", ())
    ):
        config, plan = shared_memory_architecture(source, caches, kv_heads, binding)
        with torch.device("meta"):
            model = SpeckForCausalLM(config)
        parameters = model.parameter_count()
        flops = model.flops_per_token(sequence_length)
        residual = plan["unmatched_parameters_in_feed_forward"]
        key_norms = plan["readers"] * source_head_norm(source)
        if parameters - baseline_parameters != -(residual + key_norms):
            raise ValueError(f"reader attention arm {name} has unexplained parameter accounting")
        if flops - baseline_flops != -6 * residual:
            raise ValueError(f"reader attention arm {name} has unexplained FLOPs accounting")
        matched = residual == 0
        state = state_bytes(config)
        variants[name] = config
        summary[name] = {
            **plan,
            "parameters": parameters,
            "parameter_delta_versus_source": parameters - baseline_parameters,
            "flops_per_token": flops,
            "flops_delta_versus_source": flops - baseline_flops,
            "matrix_parameter_and_flop_matched": matched,
            "unmatched_key_norm_parameters": key_norms,
            "resident_state_bytes": state,
            "resident_state_fraction_of_source": {
                length: report["total_bytes"] / baseline_state[length]["total_bytes"]
                for length, report in state.items()
            },
        }
    contract = {
        "format": "speck_reader_attention_staircase",
        "format_version": 1,
        "comparison": (
            "identical depth, attention placement, and read count with a varying number of "
            "shared key-value caches"
        ),
        "mechanism": "speck_reader_attention",
        "source_experiment": str(source_dir),
        "source_parameters": baseline_parameters,
        "source_flops_per_token": baseline_flops,
        "source_resident_state_bytes": baseline_state,
        "sequence_length": sequence_length,
        "train_tokens": train_tokens,
        "seed": seed,
        "variants": summary,
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
                "seed": seed,
                "train_tokens": train_tokens,
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
                    json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
        (building / "staircase.json").write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(building, output)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return contract


def source_head_norm(source):
    """Return the key-normalization parameter count that one reader layer removes."""

    for invocation in source.execution_plan:
        for stage in invocation.block.stages:
            for branch in stage.branches:
                if isinstance(branch, AttentionSpec):
                    return branch.head_dim
    raise ValueError("the source architecture has no attention layers")


def main(argv=None):
    args = arguments(argv)
    contract = prepare(args)
    print(f"Prepared {len(contract['variants'])} reader attention arms under {args.output_dir}")


if __name__ == "__main__":
    main()
