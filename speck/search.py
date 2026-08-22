"""deterministic architecture search mechanics."""

import hashlib
import json
import math
import os
import random
import re
import statistics
import time
from datetime import datetime, timezone
from dataclasses import dataclass, replace
from pathlib import Path

from speck.architecture import (
    ArchitectureConfig,
    AttentionSpec,
    BlockConfig,
    BlockGroup,
    GatedCausalConvSpec,
    StageConfig,
    SwiGLUSpec,
)


MUTATIONS = (
    "insert_block",
    "delete_block",
    "duplicate_independent_block",
    "move_block",
    "change_block_width",
    "change_embedding_width",
    "replace_mixer",
    "toggle_attention_scope",
    "change_sliding_window",
    "change_attention_head_dimension",
    "change_kv_heads",
    "change_convolution_width",
    "change_convolution_kernel",
    "toggle_swiglu",
    "change_swiglu_expansion",
    "change_shared_repetition",
)


class InapplicableMutation(ValueError):
    pass


def _copy_json(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _probabilities(values, name):
    if not isinstance(values, dict) or not values:
        raise ValueError(f"{name} must be a nonempty probability table")
    probabilities = {str(key): float(value) for key, value in values.items()}
    if any(value < 0 or not math.isfinite(value) for value in probabilities.values()):
        raise ValueError(f"{name} contains an invalid probability")
    if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-9):
        raise ValueError(f"{name} probabilities must sum to one")
    return probabilities


@dataclass(frozen=True)
class SearchSettings:
    values: dict

    @classmethod
    def from_dict(cls, value):
        values = _copy_json(value)
        required = {
            "format_version",
            "seed",
            "generation_size",
            "controlled_mutations",
            "random_immigrants",
            "mutation_attempts",
            "random_attempts",
            "parameter_bounds",
            "logical_depth_bounds",
            "baseline_normalization",
            "training",
            "evaluation",
            "rungs",
            "final_tokens",
            "widths",
            "head_dimensions",
            "kv_heads",
            "sliding_windows",
            "convolution_ratios",
            "convolution_kernels",
            "swiglu_ratios",
            "repeat_counts",
            "mutation_probabilities",
            "depth_probabilities",
            "embedding_width_probabilities",
            "block_width_probabilities",
            "mixer_probabilities",
            "swiglu_probabilities",
            "attention_scope_probabilities",
            "attention_head_dimension_probabilities",
            "kv_head_probabilities",
            "convolution_ratio_probabilities",
            "convolution_kernel_probabilities",
            "swiglu_ratio_probabilities",
            "repeat_count_probabilities",
            "parent_lane_probabilities",
            "profile",
            "final_profile",
            "final_seed_offset",
        }
        missing = sorted(required - values.keys())
        if missing:
            raise ValueError(f"search settings are missing: {', '.join(missing)}")
        if values["format_version"] != 1:
            raise ValueError("unsupported search settings format")

        positive = (
            "generation_size",
            "mutation_attempts",
            "random_attempts",
            "final_tokens",
        )
        if any(int(values[key]) < 1 for key in positive):
            raise ValueError("search counts and horizons must be positive")
        if not 0 <= int(values["random_immigrants"]) < int(values["generation_size"]):
            raise ValueError("random immigrants must fit in a generation")
        if len(values["controlled_mutations"]) + 1 > int(values["generation_size"]):
            raise ValueError("controlled mutations must fit in generation zero")
        if any(name not in MUTATIONS for name in values["controlled_mutations"]):
            raise ValueError("controlled mutations contain an unknown mutation")

        for key in (
            "widths",
            "head_dimensions",
            "kv_heads",
            "sliding_windows",
            "convolution_ratios",
            "convolution_kernels",
            "swiglu_ratios",
            "repeat_counts",
            "rungs",
        ):
            sequence = values[key]
            if not sequence or sequence != sorted(set(sequence)) or any(item <= 0 for item in sequence):
                raise ValueError(f"{key} must contain sorted unique positive values")

        for key in ("parameter_bounds", "logical_depth_bounds"):
            bounds = values[key]
            if len(bounds) != 2 or bounds[0] < 1 or bounds[0] > bounds[1]:
                raise ValueError(f"{key} must contain valid lower and upper bounds")
        if values["rungs"][-1] >= values["final_tokens"]:
            raise ValueError("final tokens must exceed the final search rung")

        training = values["training"]
        training_required = {
            "sequence_length",
            "device_batch_size",
            "batch_tokens",
            "optimizer",
            "learning_rate",
            "weight_decay",
            "gradient_clip",
            "warmup_tokens",
            "schedule_tokens",
            "minimum_learning_rate_scale",
            "checkpoints",
        }
        if training_required - training.keys():
            raise ValueError("training settings are incomplete")
        micro_tokens = training["sequence_length"] * training["device_batch_size"]
        if micro_tokens < 1 or training["batch_tokens"] % micro_tokens:
            raise ValueError("training batch tokens must divide into whole micro batches")
        if any(tokens % training["batch_tokens"] for tokens in training["checkpoints"]):
            raise ValueError("training checkpoints must align with optimizer batches")
        if any(rung not in training["checkpoints"] for rung in values["rungs"]):
            raise ValueError("every rung must be a training checkpoint")
        if training["schedule_tokens"] != values["final_tokens"]:
            raise ValueError("training schedule and final horizons must match")

        evaluation = values["evaluation"]
        if evaluation.get("monitor_offset") != 0:
            raise ValueError("the monitor slice must begin at zero")
        if evaluation.get("final_offset") != evaluation.get("monitor_tokens"):
            raise ValueError("the final slice must immediately follow the monitor slice")
        if evaluation.get("monitor_tokens", 0) < 1 or evaluation.get("final_tokens", 0) < 1:
            raise ValueError("evaluation slices must contain targets")

        probability_keys = (
            "mutation_probabilities",
            "depth_probabilities",
            "embedding_width_probabilities",
            "block_width_probabilities",
            "mixer_probabilities",
            "swiglu_probabilities",
            "attention_scope_probabilities",
            "attention_head_dimension_probabilities",
            "kv_head_probabilities",
            "convolution_ratio_probabilities",
            "convolution_kernel_probabilities",
            "swiglu_ratio_probabilities",
            "repeat_count_probabilities",
            "parent_lane_probabilities",
        )
        for key in probability_keys:
            values[key] = _probabilities(values[key], key)

        def ordered(key, order):
            if set(values[key]) != {str(item) for item in order}:
                raise ValueError(f"{key} does not match its search choices")
            values[key] = {
                str(item): values[key][str(item)]
                for item in order
            }

        ordered("mutation_probabilities", MUTATIONS)
        ordered("depth_probabilities", sorted(int(key) for key in values["depth_probabilities"]))
        ordered(
            "embedding_width_probabilities",
            sorted(int(key) for key in values["embedding_width_probabilities"]),
        )
        ordered("block_width_probabilities", ("same", "narrower", "wider"))
        ordered("mixer_probabilities", ("attention", "convolution", "none"))
        ordered("swiglu_probabilities", ("enabled", "disabled"))
        ordered("attention_scope_probabilities", ("global", "sliding"))
        ordered("attention_head_dimension_probabilities", values["head_dimensions"])
        ordered("kv_head_probabilities", values["kv_heads"])
        ordered("convolution_ratio_probabilities", values["convolution_ratios"])
        ordered("convolution_kernel_probabilities", values["convolution_kernels"])
        ordered("swiglu_ratio_probabilities", values["swiglu_ratios"])
        ordered("repeat_count_probabilities", values["repeat_counts"])
        ordered("parent_lane_probabilities", ("quality", "balanced", "efficiency"))
        return cls(values)

    def settings(self):
        return _copy_json(self.values)

    def __getitem__(self, key):
        return self.values[key]


def load_search_settings(path):
    return SearchSettings.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def derived_seed(seed, *parts):
    payload = ":".join((str(seed), *(str(part) for part in parts))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _rng(value):
    return value if isinstance(value, random.Random) else random.Random(value)


def _weighted_choice(rng, probabilities, allowed=None):
    items = [
        (key, probability)
        for key, probability in probabilities.items()
        if allowed is None or key in allowed
    ]
    total = sum(probability for _, probability in items)
    if total <= 0:
        raise ValueError("no weighted choices are available")
    target = rng.random() * total
    cumulative = 0.0
    for key, probability in items:
        cumulative += probability
        if target < cumulative:
            return key
    return items[-1][0]


def _numeric_choice(rng, probabilities, cast, allowed=None):
    allowed_keys = None if allowed is None else {str(value) for value in allowed}
    return cast(_weighted_choice(rng, probabilities, allowed_keys))


def _parts(block):
    if any(len(stage.branches) != 1 for stage in block.stages):
        raise ValueError("search stages must contain one operation")
    operations = tuple(stage.branches[0] for stage in block.stages)
    if not operations or len(operations) > 2:
        raise ValueError("search blocks contain at most a mixer and swiglu")
    if len(operations) == 1:
        mixer = None if isinstance(operations[0], SwiGLUSpec) else operations[0]
        swiglu = operations[0] if isinstance(operations[0], SwiGLUSpec) else None
    else:
        mixer, swiglu = operations
    if mixer is not None and not isinstance(
        mixer, (AttentionSpec, GatedCausalConvSpec)
    ):
        raise ValueError("search blocks contain one supported mixer")
    if swiglu is not None and not isinstance(swiglu, SwiGLUSpec):
        raise ValueError("swiglu must follow the mixer")
    return mixer, swiglu


def _block(hidden_size, mixer, swiglu):
    operations = tuple(operation for operation in (mixer, swiglu) if operation is not None)
    return BlockConfig(hidden_size, tuple(StageConfig((operation,)) for operation in operations))


def _closest(values, target):
    return min(values, key=lambda value: (abs(value - target), value))


def normalize_baseline(config, settings):
    normalization = settings["baseline_normalization"]
    replacements = {
        int(source): int(target)
        for source, target in normalization.get("width_replacements", {}).items()
    }
    groups = []
    for group in config.blocks:
        old_width = group.block.hidden_size
        new_width = replacements.get(old_width, old_width)
        mixer, swiglu = _parts(group.block)
        if isinstance(mixer, GatedCausalConvSpec):
            ratio = _closest(settings["convolution_ratios"], mixer.inner_size / old_width)
            mixer = replace(mixer, inner_size=round(new_width * ratio))
        if swiglu is not None:
            ratio = _closest(settings["swiglu_ratios"], swiglu.intermediate_size / old_width)
            swiglu = replace(swiglu, intermediate_size=round(new_width * ratio))
        normalized = BlockGroup(
            _block(new_width, mixer, swiglu),
            repeat=group.repeat,
            weight_sharing=group.weight_sharing,
        )
        if (
            normalization.get("expand_unshared_repetitions")
            and normalized.repeat > 1
            and normalized.weight_sharing == "none"
        ):
            groups.extend(BlockGroup(normalized.block) for _ in range(normalized.repeat))
        else:
            groups.append(
                replace(
                    normalized,
                    weight_sharing="all" if normalized.repeat > 1 else "none",
                )
            )
    embedding_size = replacements.get(config.embedding_size, config.embedding_size)
    normalized = replace(
        config,
        blocks=tuple(groups),
        embedding_size=embedding_size,
        expected_parameters=None,
    )
    validate_architecture(normalized, settings)
    return normalized


def parameter_count(config):
    total = config.vocab_size * config.embedding_size
    counted = set()
    input_size = config.embedding_size
    for invocation in config.execution_plan:
        hidden_size = invocation.block.hidden_size
        if input_size != hidden_size:
            total += input_size * hidden_size
        if invocation.weight_key not in counted:
            for stage in invocation.block.stages:
                for operation in stage.branches:
                    total += hidden_size
                    if isinstance(operation, AttentionSpec):
                        kv_size = operation.num_key_value_heads * operation.head_dim
                        total += 2 * hidden_size * hidden_size
                        total += 2 * hidden_size * kv_size
                        total += 2 * operation.head_dim
                    elif isinstance(operation, GatedCausalConvSpec):
                        total += 4 * hidden_size * operation.inner_size
                        total += operation.inner_size * operation.kernel_size
                    else:
                        total += 3 * hidden_size * operation.intermediate_size
            counted.add(invocation.weight_key)
        input_size = hidden_size
    total += input_size
    if input_size != config.embedding_size:
        total += input_size * config.embedding_size
    return total


def sequence_state_bytes(config, length, batch_size=1, element_size=2):
    total = 0
    for invocation in config.execution_plan:
        for stage in invocation.block.stages:
            for operation in stage.branches:
                if isinstance(operation, AttentionSpec):
                    capacity = length
                    if operation.scope == "sliding":
                        capacity = min(length, operation.window_size)
                    total += (
                        2
                        * batch_size
                        * operation.num_key_value_heads
                        * capacity
                        * operation.head_dim
                        * element_size
                    )
                elif isinstance(operation, GatedCausalConvSpec):
                    total += (
                        batch_size
                        * operation.inner_size
                        * (operation.kernel_size - 1)
                        * element_size
                    )
    return total


def flops_per_token(config, sequence_length):
    linear = config.vocab_size * config.embedding_size
    attention = 0
    input_size = config.embedding_size
    for invocation in config.execution_plan:
        hidden_size = invocation.block.hidden_size
        if input_size != hidden_size:
            linear += input_size * hidden_size
        for stage in invocation.block.stages:
            for operation in stage.branches:
                if isinstance(operation, AttentionSpec):
                    kv_size = operation.num_key_value_heads * operation.head_dim
                    linear += 2 * hidden_size * hidden_size + 2 * hidden_size * kv_size
                    context = sequence_length
                    if operation.scope == "sliding":
                        context = min(sequence_length, operation.window_size)
                    attention += 12 * context * hidden_size
                elif isinstance(operation, GatedCausalConvSpec):
                    linear += 4 * hidden_size * operation.inner_size
                    linear += operation.inner_size * operation.kernel_size
                else:
                    linear += 3 * hidden_size * operation.intermediate_size
        input_size = hidden_size
    if input_size != config.embedding_size:
        linear += input_size * config.embedding_size
    return 6 * linear + attention


def architecture_metrics(config, settings):
    executions = {"attention": 0, "convolution": 0, "swiglu": 0}
    for invocation in config.execution_plan:
        for stage in invocation.block.stages:
            for operation in stage.branches:
                if isinstance(operation, AttentionSpec):
                    executions["attention"] += 1
                elif isinstance(operation, GatedCausalConvSpec):
                    executions["convolution"] += 1
                else:
                    executions["swiglu"] += 1
    parameters = parameter_count(config)
    sequence_length = settings["training"]["sequence_length"]
    return {
        "parameters": parameters,
        "weight_bytes": parameters * 4,
        "flops_per_token": flops_per_token(config, sequence_length),
        "state_bytes": {
            str(length): sequence_state_bytes(config, length)
            for length in (512, 2048, 4096)
        },
        "logical_depth": config.logical_depth,
        "unique_parameter_blocks": config.unique_parameter_blocks,
        "executions": executions,
    }


def validate_architecture(config, settings):
    if config.embedding_size not in settings["widths"]:
        raise ValueError("embedding width is outside the search space")
    minimum_depth, maximum_depth = settings["logical_depth_bounds"]
    if not minimum_depth <= config.logical_depth <= maximum_depth:
        raise ValueError("logical depth is outside the search bounds")
    for group in config.blocks:
        if group.repeat not in settings["repeat_counts"]:
            raise ValueError("shared repetition is outside the search space")
        expected_sharing = "all" if group.repeat > 1 else "none"
        if group.weight_sharing != expected_sharing:
            raise ValueError("search repetitions must use canonical sharing")
        block = group.block
        if block.hidden_size not in settings["widths"]:
            raise ValueError("block width is outside the search space")
        mixer, swiglu = _parts(block)
        if isinstance(mixer, AttentionSpec):
            if mixer.head_dim not in settings["head_dimensions"]:
                raise ValueError("attention head dimension is outside the search space")
            if mixer.num_key_value_heads not in settings["kv_heads"]:
                raise ValueError("attention kv heads are outside the search space")
            if mixer.scope == "sliding" and mixer.window_size not in settings["sliding_windows"]:
                raise ValueError("sliding window is outside the search space")
        elif isinstance(mixer, GatedCausalConvSpec):
            ratio = mixer.inner_size / block.hidden_size
            if not any(math.isclose(ratio, value) for value in settings["convolution_ratios"]):
                raise ValueError("convolution ratio is outside the search space")
            if mixer.kernel_size not in settings["convolution_kernels"]:
                raise ValueError("convolution kernel is outside the search space")
        if swiglu is not None:
            ratio = swiglu.intermediate_size / block.hidden_size
            if not any(math.isclose(ratio, value) for value in settings["swiglu_ratios"]):
                raise ValueError("swiglu ratio is outside the search space")
        if mixer is None and swiglu is None:
            raise ValueError("a search block requires a mixer or swiglu")
    parameters = parameter_count(config)
    lower, upper = settings["parameter_bounds"]
    if not lower <= parameters <= upper:
        raise ValueError("parameter count is outside the search bounds")
    return architecture_metrics(config, settings)


def _step(values, current, rng):
    index = values.index(current)
    choices = []
    if index:
        choices.append(values[index - 1])
    if index + 1 < len(values):
        choices.append(values[index + 1])
    if not choices:
        raise InapplicableMutation("numeric value has no adjacent choice")
    return rng.choice(choices)


def _random_width(previous, settings, rng):
    direction = _weighted_choice(rng, settings["block_width_probabilities"])
    widths = settings["widths"]
    index = widths.index(previous)
    if direction == "narrower" and index:
        return widths[index - 1]
    if direction == "wider" and index + 1 < len(widths):
        return widths[index + 1]
    return previous


def _valid_kv_heads(width, head_dim, settings):
    if width % head_dim:
        return []
    query_heads = width // head_dim
    return [heads for heads in settings["kv_heads"] if query_heads % heads == 0]


def _random_group(width, remaining_depth, settings, rng):
    mixer_name = _weighted_choice(rng, settings["mixer_probabilities"])
    swiglu_enabled = _weighted_choice(rng, settings["swiglu_probabilities"]) == "enabled"
    if mixer_name == "none":
        swiglu_enabled = True

    mixer = None
    if mixer_name == "attention":
        dimensions = [
            value
            for value in settings["head_dimensions"]
            if _valid_kv_heads(width, value, settings)
        ]
        head_dim = _numeric_choice(
            rng,
            settings["attention_head_dimension_probabilities"],
            int,
            dimensions,
        )
        kv_heads = _numeric_choice(
            rng,
            settings["kv_head_probabilities"],
            int,
            _valid_kv_heads(width, head_dim, settings),
        )
        scope = _weighted_choice(rng, settings["attention_scope_probabilities"])
        window = rng.choice(settings["sliding_windows"]) if scope == "sliding" else None
        mixer = AttentionSpec(head_dim, kv_heads, scope, window)
    elif mixer_name == "convolution":
        ratio = _numeric_choice(
            rng, settings["convolution_ratio_probabilities"], float
        )
        kernel = _numeric_choice(
            rng, settings["convolution_kernel_probabilities"], int
        )
        mixer = GatedCausalConvSpec(round(width * ratio), kernel)

    swiglu = None
    if swiglu_enabled:
        ratio = _numeric_choice(rng, settings["swiglu_ratio_probabilities"], int)
        swiglu = SwiGLUSpec(width * ratio)
    repeat = _numeric_choice(
        rng,
        settings["repeat_count_probabilities"],
        int,
        [value for value in settings["repeat_counts"] if value <= remaining_depth],
    )
    return BlockGroup(
        _block(width, mixer, swiglu),
        repeat=repeat,
        weight_sharing="all" if repeat > 1 else "none",
    )


def random_architecture(template, settings, seed, attempts=None):
    rng = _rng(seed)
    attempts = attempts or settings["random_attempts"]
    last_error = None
    for _ in range(attempts):
        depth = _numeric_choice(rng, settings["depth_probabilities"], int)
        embedding_size = _numeric_choice(
            rng, settings["embedding_width_probabilities"], int
        )
        groups = []
        previous = embedding_size
        remaining = depth
        while remaining:
            width = _random_width(previous, settings, rng)
            group = _random_group(width, remaining, settings, rng)
            groups.append(group)
            previous = width
            remaining -= group.repeat
        config = replace(
            template,
            blocks=tuple(groups),
            embedding_size=embedding_size,
            expected_parameters=None,
        )
        try:
            validate_architecture(config, settings)
            return config
        except ValueError as error:
            last_error = error
    raise RuntimeError("failed to generate a feasible random architecture") from last_error


def _replace_group(config, index, group):
    groups = list(config.blocks)
    groups[index] = group
    return replace(config, blocks=tuple(groups), expected_parameters=None)


def _mutation_insert(config, settings, rng):
    if config.logical_depth >= settings["logical_depth_bounds"][1]:
        raise InapplicableMutation("architecture is at maximum depth")
    index = rng.randrange(len(config.blocks) + 1)
    previous = config.embedding_size if index == 0 else config.blocks[index - 1].block.hidden_size
    width = _random_width(previous, settings, rng)
    group = _random_group(width, 1, settings, rng)
    groups = list(config.blocks)
    groups.insert(index, group)
    return replace(config, blocks=tuple(groups), expected_parameters=None), {"index": index}


def _mutation_delete(config, settings, rng):
    minimum = settings["logical_depth_bounds"][0]
    choices = [
        index
        for index, group in enumerate(config.blocks)
        if config.logical_depth - group.repeat >= minimum
    ]
    if not choices:
        raise InapplicableMutation("no block can be deleted within the depth bound")
    index = rng.choice(choices)
    groups = list(config.blocks)
    groups.pop(index)
    return replace(config, blocks=tuple(groups), expected_parameters=None), {"index": index}


def _mutation_duplicate(config, settings, rng):
    maximum = settings["logical_depth_bounds"][1]
    choices = [
        index
        for index, group in enumerate(config.blocks)
        if config.logical_depth + group.repeat <= maximum
    ]
    if not choices:
        raise InapplicableMutation("no block can be duplicated within the depth bound")
    source = rng.choice(choices)
    destination = rng.randrange(len(config.blocks) + 1)
    groups = list(config.blocks)
    groups.insert(destination, config.blocks[source])
    return replace(config, blocks=tuple(groups), expected_parameters=None), {
        "source": source,
        "destination": destination,
    }


def _mutation_move(config, settings, rng):
    if len(config.blocks) < 2:
        raise InapplicableMutation("moving a block requires two groups")
    source = rng.randrange(len(config.blocks))
    destinations = [index for index in range(len(config.blocks)) if index != source]
    destination = rng.choice(destinations)
    groups = list(config.blocks)
    group = groups.pop(source)
    groups.insert(destination, group)
    return replace(config, blocks=tuple(groups), expected_parameters=None), {
        "source": source,
        "destination": destination,
    }


def _mutation_block_width(config, settings, rng):
    choices = [
        index
        for index, group in enumerate(config.blocks)
        if len(settings["widths"]) > 1
    ]
    if not choices:
        raise InapplicableMutation("no block width can change")
    index = rng.choice(choices)
    group = config.blocks[index]
    old_width = group.block.hidden_size
    new_width = _step(settings["widths"], old_width, rng)
    mixer, swiglu = _parts(group.block)
    if isinstance(mixer, GatedCausalConvSpec):
        ratio = mixer.inner_size / old_width
        mixer = replace(mixer, inner_size=round(new_width * ratio))
    if swiglu is not None:
        ratio = swiglu.intermediate_size / old_width
        swiglu = replace(swiglu, intermediate_size=round(new_width * ratio))
    changed = replace(group, block=_block(new_width, mixer, swiglu))
    return _replace_group(config, index, changed), {
        "index": index,
        "from": old_width,
        "to": new_width,
    }


def _mutation_embedding_width(config, settings, rng):
    width = _step(settings["widths"], config.embedding_size, rng)
    return replace(config, embedding_size=width, expected_parameters=None), {
        "from": config.embedding_size,
        "to": width,
    }


def _mutation_replace_mixer(config, settings, rng):
    index = rng.randrange(len(config.blocks))
    group = config.blocks[index]
    mixer, swiglu = _parts(group.block)
    current = (
        "attention"
        if isinstance(mixer, AttentionSpec)
        else "convolution"
        if isinstance(mixer, GatedCausalConvSpec)
        else "none"
    )
    choices = [name for name in ("attention", "convolution", "none") if name != current]
    if swiglu is None:
        choices.remove("none")
    replacement_name = rng.choice(choices)
    replacement = None
    if replacement_name == "attention":
        dimensions = [
            value
            for value in settings["head_dimensions"]
            if _valid_kv_heads(group.block.hidden_size, value, settings)
        ]
        if not dimensions:
            raise InapplicableMutation("block width has no valid attention head dimension")
        replacement = AttentionSpec(dimensions[0], 1)
    elif replacement_name == "convolution":
        ratio = 1.0 if 1.0 in settings["convolution_ratios"] else settings["convolution_ratios"][0]
        replacement = GatedCausalConvSpec(
            round(group.block.hidden_size * ratio),
            settings["convolution_kernels"][0],
        )
    changed = replace(group, block=_block(group.block.hidden_size, replacement, swiglu))
    return _replace_group(config, index, changed), {
        "index": index,
        "from": current,
        "to": replacement_name,
    }


def _groups_with(config, operation_type):
    choices = []
    for index, group in enumerate(config.blocks):
        mixer, swiglu = _parts(group.block)
        operation = swiglu if operation_type is SwiGLUSpec else mixer
        if isinstance(operation, operation_type):
            choices.append((index, group, mixer, swiglu, operation))
    return choices


def _mutation_attention_scope(config, settings, rng):
    choices = _groups_with(config, AttentionSpec)
    if not choices:
        raise InapplicableMutation("architecture has no attention")
    index, group, mixer, swiglu, attention = rng.choice(choices)
    if attention.scope == "global":
        changed_attention = replace(
            attention,
            scope="sliding",
            window_size=settings["sliding_windows"][0],
        )
    else:
        changed_attention = replace(attention, scope="global", window_size=None)
    changed = replace(group, block=_block(group.block.hidden_size, changed_attention, swiglu))
    return _replace_group(config, index, changed), {
        "index": index,
        "from": attention.scope,
        "to": changed_attention.scope,
    }


def _mutation_sliding_window(config, settings, rng):
    choices = [
        choice
        for choice in _groups_with(config, AttentionSpec)
        if choice[4].scope == "sliding" and len(settings["sliding_windows"]) > 1
    ]
    if not choices:
        raise InapplicableMutation("architecture has no mutable sliding attention")
    index, group, mixer, swiglu, attention = rng.choice(choices)
    window = _step(settings["sliding_windows"], attention.window_size, rng)
    changed = replace(
        group,
        block=_block(group.block.hidden_size, replace(attention, window_size=window), swiglu),
    )
    return _replace_group(config, index, changed), {
        "index": index,
        "from": attention.window_size,
        "to": window,
    }


def _mutation_head_dimension(config, settings, rng):
    choices = _groups_with(config, AttentionSpec)
    if not choices or len(settings["head_dimensions"]) < 2:
        raise InapplicableMutation("architecture has no mutable attention head dimension")
    index, group, mixer, swiglu, attention = rng.choice(choices)
    head_dim = _step(settings["head_dimensions"], attention.head_dim, rng)
    changed = replace(
        group,
        block=_block(group.block.hidden_size, replace(attention, head_dim=head_dim), swiglu),
    )
    return _replace_group(config, index, changed), {
        "index": index,
        "from": attention.head_dim,
        "to": head_dim,
    }


def _mutation_kv_heads(config, settings, rng):
    choices = _groups_with(config, AttentionSpec)
    if not choices or len(settings["kv_heads"]) < 2:
        raise InapplicableMutation("architecture has no mutable kv head count")
    index, group, mixer, swiglu, attention = rng.choice(choices)
    heads = _step(settings["kv_heads"], attention.num_key_value_heads, rng)
    changed = replace(
        group,
        block=_block(
            group.block.hidden_size,
            replace(attention, num_key_value_heads=heads),
            swiglu,
        ),
    )
    return _replace_group(config, index, changed), {
        "index": index,
        "from": attention.num_key_value_heads,
        "to": heads,
    }


def _mutation_convolution_width(config, settings, rng):
    choices = _groups_with(config, GatedCausalConvSpec)
    if not choices or len(settings["convolution_ratios"]) < 2:
        raise InapplicableMutation("architecture has no mutable convolution width")
    index, group, mixer, swiglu, convolution = rng.choice(choices)
    old_ratio = convolution.inner_size / group.block.hidden_size
    ratio = _step(settings["convolution_ratios"], old_ratio, rng)
    changed_mixer = replace(
        convolution,
        inner_size=round(group.block.hidden_size * ratio),
    )
    changed = replace(group, block=_block(group.block.hidden_size, changed_mixer, swiglu))
    return _replace_group(config, index, changed), {
        "index": index,
        "from": old_ratio,
        "to": ratio,
    }


def _mutation_convolution_kernel(config, settings, rng):
    choices = _groups_with(config, GatedCausalConvSpec)
    if not choices or len(settings["convolution_kernels"]) < 2:
        raise InapplicableMutation("architecture has no mutable convolution kernel")
    index, group, mixer, swiglu, convolution = rng.choice(choices)
    kernel = _step(settings["convolution_kernels"], convolution.kernel_size, rng)
    changed = replace(
        group,
        block=_block(
            group.block.hidden_size,
            replace(convolution, kernel_size=kernel),
            swiglu,
        ),
    )
    return _replace_group(config, index, changed), {
        "index": index,
        "from": convolution.kernel_size,
        "to": kernel,
    }


def _mutation_toggle_swiglu(config, settings, rng):
    choices = []
    for index, group in enumerate(config.blocks):
        mixer, swiglu = _parts(group.block)
        if swiglu is None or mixer is not None:
            choices.append((index, group, mixer, swiglu))
    if not choices:
        raise InapplicableMutation("no swiglu can be toggled")
    index, group, mixer, swiglu = rng.choice(choices)
    if swiglu is None:
        ratio = 3 if 3 in settings["swiglu_ratios"] else settings["swiglu_ratios"][0]
        changed_swiglu = SwiGLUSpec(group.block.hidden_size * ratio)
    else:
        changed_swiglu = None
    changed = replace(group, block=_block(group.block.hidden_size, mixer, changed_swiglu))
    return _replace_group(config, index, changed), {
        "index": index,
        "enabled": changed_swiglu is not None,
    }


def _mutation_swiglu_expansion(config, settings, rng):
    choices = _groups_with(config, SwiGLUSpec)
    if not choices or len(settings["swiglu_ratios"]) < 2:
        raise InapplicableMutation("architecture has no mutable swiglu expansion")
    index, group, mixer, swiglu, operation = rng.choice(choices)
    old_ratio = operation.intermediate_size / group.block.hidden_size
    ratio = _step(settings["swiglu_ratios"], old_ratio, rng)
    changed_swiglu = replace(
        operation,
        intermediate_size=group.block.hidden_size * ratio,
    )
    changed = replace(group, block=_block(group.block.hidden_size, mixer, changed_swiglu))
    return _replace_group(config, index, changed), {
        "index": index,
        "from": old_ratio,
        "to": ratio,
    }


def _mutation_repetition(config, settings, rng):
    choices = []
    minimum, maximum = settings["logical_depth_bounds"]
    for index, group in enumerate(config.blocks):
        repeat_index = settings["repeat_counts"].index(group.repeat)
        possible = []
        if repeat_index:
            possible.append(settings["repeat_counts"][repeat_index - 1])
        if repeat_index + 1 < len(settings["repeat_counts"]):
            possible.append(settings["repeat_counts"][repeat_index + 1])
        possible = [
            repeat
            for repeat in possible
            if minimum <= config.logical_depth - group.repeat + repeat <= maximum
        ]
        if possible:
            choices.append((index, group, possible))
    if not choices:
        raise InapplicableMutation("no shared repetition can change within the depth bounds")
    index, group, possible = rng.choice(choices)
    repeat = rng.choice(possible)
    changed = replace(group, repeat=repeat, weight_sharing="all" if repeat > 1 else "none")
    return _replace_group(config, index, changed), {
        "index": index,
        "from": group.repeat,
        "to": repeat,
    }


_MUTATION_FUNCTIONS = {
    "insert_block": _mutation_insert,
    "delete_block": _mutation_delete,
    "duplicate_independent_block": _mutation_duplicate,
    "move_block": _mutation_move,
    "change_block_width": _mutation_block_width,
    "change_embedding_width": _mutation_embedding_width,
    "replace_mixer": _mutation_replace_mixer,
    "toggle_attention_scope": _mutation_attention_scope,
    "change_sliding_window": _mutation_sliding_window,
    "change_attention_head_dimension": _mutation_head_dimension,
    "change_kv_heads": _mutation_kv_heads,
    "change_convolution_width": _mutation_convolution_width,
    "change_convolution_kernel": _mutation_convolution_kernel,
    "toggle_swiglu": _mutation_toggle_swiglu,
    "change_swiglu_expansion": _mutation_swiglu_expansion,
    "change_shared_repetition": _mutation_repetition,
}


@dataclass(frozen=True)
class MutationResult:
    architecture: ArchitectureConfig
    name: str
    details: dict


def mutate_architecture(parent, settings, seed, mutation=None, attempts=None):
    if mutation is not None and mutation not in MUTATIONS:
        raise ValueError(f"unknown mutation: {mutation}")
    rng = _rng(seed)
    attempts = attempts or settings["mutation_attempts"]
    last_error = None
    for _ in range(attempts):
        name = mutation or _weighted_choice(rng, settings["mutation_probabilities"])
        try:
            architecture, details = _MUTATION_FUNCTIONS[name](parent, settings, rng)
            if architecture.digest == parent.digest:
                raise InapplicableMutation("mutation did not change architecture identity")
            validate_architecture(architecture, settings)
            return MutationResult(architecture, name, details)
        except (InapplicableMutation, ValueError) as error:
            last_error = error
    qualifier = mutation or "sampled"
    raise InapplicableMutation(f"failed to apply {qualifier} mutation") from last_error


@dataclass(frozen=True)
class CandidatePlan:
    architecture: ArchitectureConfig
    parent_digest: str | None
    mutation: dict | None


def _unique_random(template, settings, rng, seen):
    last_error = None
    for _ in range(settings["random_attempts"]):
        try:
            architecture = random_architecture(template, settings, rng)
        except RuntimeError as error:
            last_error = error
            continue
        if architecture.digest not in seen:
            seen.add(architecture.digest)
            return architecture
    raise RuntimeError("failed to generate a unique random architecture") from last_error


def initial_generation(baseline, settings, existing_digests=()):
    rng = random.Random(derived_seed(settings["seed"], "generation", 0))
    normalized = normalize_baseline(baseline, settings)
    seen = set(existing_digests)
    if normalized.digest in seen:
        raise RuntimeError("normalized baseline duplicates an existing candidate")
    seen.add(normalized.digest)
    plans = [CandidatePlan(normalized, None, None)]
    for name in settings["controlled_mutations"]:
        result = None
        for _ in range(settings["mutation_attempts"]):
            try:
                candidate = mutate_architecture(normalized, settings, rng, name)
            except InapplicableMutation:
                continue
            if candidate.architecture.digest not in seen:
                result = candidate
                break
        if result is None:
            raise RuntimeError(f"failed to create unique controlled mutation: {name}")
        seen.add(result.architecture.digest)
        plans.append(
            CandidatePlan(
                result.architecture,
                normalized.digest,
                {"name": result.name, **result.details},
            )
        )
    while len(plans) < settings["generation_size"]:
        architecture = _unique_random(normalized, settings, rng, seen)
        plans.append(CandidatePlan(architecture, None, {"name": "random"}))
    return tuple(plans)


def _regression(points):
    if len(points) < 3:
        raise ValueError("learning-curve regression requires three points")
    selected = sorted(points, key=lambda point: point["tokens"])[-3:]
    x = [math.log2(point["tokens"]) for point in selected]
    y = [float(point["nll"]) for point in selected]
    if any(not math.isfinite(value) for value in y):
        raise ValueError("learning-curve nll values must be finite")
    x_mean = statistics.mean(x)
    y_mean = statistics.mean(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    slope = sum((x_value - x_mean) * (y_value - y_mean) for x_value, y_value in zip(x, y))
    slope /= denominator
    intercept = y_mean - slope * x_mean
    residual = sum(
        (y_value - (intercept + slope * x_value)) ** 2
        for x_value, y_value in zip(x, y)
    )
    total = sum((value - y_mean) ** 2 for value in y)
    r_squared = 0.0 if total == 0 else max(0.0, min(1.0, 1 - residual / total))
    return {
        "measured_slope": slope,
        "r_squared": r_squared,
        "current_tokens": selected[-1]["tokens"],
        "current_nll": y[-1],
        "fit_tokens": [point["tokens"] for point in selected],
    }


def project_learning_curve(points, next_horizon, median_slope):
    estimate = _regression(points)
    effective_slope = (
        estimate["r_squared"] * estimate["measured_slope"]
        + (1 - estimate["r_squared"]) * median_slope
    )
    projected_nll = estimate["current_nll"] + effective_slope * math.log2(
        next_horizon / estimate["current_tokens"]
    )
    return {
        **estimate,
        "median_slope": median_slope,
        "effective_slope": effective_slope,
        "projected_tokens": next_horizon,
        "projected_nll": projected_nll,
    }


def percentile_ranks(values):
    if not values:
        return {}
    if len(values) == 1:
        return {next(iter(values)): 0.0}
    ranks = {}
    for key, value in values.items():
        lower = sum(other < value for other in values.values())
        tied = sum(other == value for other in values.values())
        ranks[key] = (lower + (tied - 1) / 2) / (len(values) - 1)
    return ranks


def _record_key(record):
    return str(record.get("candidate_id", record.get("id")))


def _metric(record, *path):
    value = record
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def score_candidates(records, next_horizon):
    scored = [_copy_json(record) for record in records]
    estimates = {}
    for record in scored:
        if record.get("status") == "failed":
            continue
        curve = record.get("nll_curve", [])
        if len(curve) >= 3:
            estimates[_record_key(record)] = _regression(curve)
    if not estimates:
        return scored
    median_slope = statistics.median(
        estimate["measured_slope"] for estimate in estimates.values()
    )
    for record in scored:
        key = _record_key(record)
        if key in estimates:
            record["forecast"] = project_learning_curve(
                record["nll_curve"], next_horizon, median_slope
            )

    current = {
        _record_key(record): record["forecast"]["current_nll"]
        for record in scored
        if "forecast" in record
    }
    projected = {
        _record_key(record): record["forecast"]["projected_nll"]
        for record in scored
        if "forecast" in record
    }
    current_ranks = percentile_ranks(current)
    projected_ranks = percentile_ranks(projected)
    for record in scored:
        key = _record_key(record)
        if key in current_ranks:
            record.setdefault("scores", {})["quality"] = (
                current_ranks[key] + projected_ranks[key]
            ) / 2

    profile_metrics = {
        "prefill_512": ("profile", "latency", "prefill_512", "p50_seconds"),
        "prefill_2048": ("profile", "latency", "prefill_2048", "p50_seconds"),
        "decode_2048": ("profile", "latency", "decode_2048", "p50_seconds"),
        "weight_bytes": ("profile", "static", "weight_bytes"),
        "state_bytes_2048": ("profile", "static", "state_bytes", "2048"),
        "peak_vram": ("profile", "memory", "peak_vram_bytes"),
    }
    ranks = {}
    for name, path in profile_metrics.items():
        values = {
            _record_key(record): value
            for record in scored
            if (value := _metric(record, *path)) is not None
        }
        ranks[name] = percentile_ranks(values)

    for record in scored:
        key = _record_key(record)
        if not all(key in ranks[name] for name in profile_metrics):
            continue
        latency = statistics.mean(
            ranks[name][key]
            for name in ("prefill_512", "prefill_2048", "decode_2048")
        )
        memory = statistics.mean(
            ranks[name][key]
            for name in ("weight_bytes", "state_bytes_2048", "peak_vram")
        )
        scores = record.setdefault("scores", {})
        scores.update(
            latency=latency,
            memory=memory,
            efficiency=(latency + memory) / 2,
        )
        if "quality" in scores:
            scores["balanced"] = (scores["quality"] + scores["efficiency"]) / 2
    return scored


def _eligible_records(records, lane):
    return [
        record
        for record in records
        if record.get("status") != "failed"
        and math.isfinite(record.get("scores", {}).get(lane, math.nan))
    ]


def lane_leaders(records):
    leaders = {}
    for lane in ("quality", "balanced", "efficiency"):
        eligible = _eligible_records(records, lane)
        if eligible:
            leaders[lane] = min(
                eligible,
                key=lambda record: (record["scores"][lane], _record_key(record)),
            )
    return leaders


def select_promotions(records, lane_counts):
    requested = sum(lane_counts.values())
    winners = []
    for lane in ("quality", "balanced", "efficiency"):
        ordered = sorted(
            _eligible_records(records, lane),
            key=lambda record: (record["scores"][lane], _record_key(record)),
        )
        winners.extend(ordered[: lane_counts.get(lane, 0)])
    selected = []
    selected_keys = set()
    for record in winners:
        key = _record_key(record)
        if key not in selected_keys:
            selected.append(record)
            selected_keys.add(key)
    balanced = sorted(
        _eligible_records(records, "balanced"),
        key=lambda record: (record["scores"]["balanced"], _record_key(record)),
    )
    for record in balanced:
        if len(selected) == requested:
            break
        key = _record_key(record)
        if key not in selected_keys:
            selected.append(record)
            selected_keys.add(key)
    if len(selected) != requested:
        raise RuntimeError("not enough eligible candidates for promotion")
    return tuple(selected)


def promotion_for_rung(records, settings, rung):
    if rung == settings["rungs"][0]:
        return select_promotions(records, {"quality": 2, "balanced": 1, "efficiency": 1})
    if rung == settings["rungs"][1]:
        return select_promotions(records, {"quality": 1, "balanced": 1, "efficiency": 1})
    raise ValueError("the final search rung is not promoted")


def _parent_scores(record, parent_rung):
    stored = record.get("scores_by_rung", {}).get(str(parent_rung))
    return stored if stored is not None else record.get("scores", {})


def select_parent(records, settings, seed):
    rng = _rng(seed)
    parent_rung = settings["rungs"][1]
    eligible = [
        record
        for record in records
        if record.get("status") != "failed"
        and record.get("trained_tokens", record.get("rung", 0)) >= parent_rung
        and all(
            math.isfinite(_parent_scores(record, parent_rung).get(lane, math.nan))
            for lane in ("quality", "balanced", "efficiency")
        )
    ]
    if not eligible:
        raise RuntimeError("the parent pool has no eligible candidates")
    lane = _weighted_choice(rng, settings["parent_lane_probabilities"])
    ordered = sorted(
        eligible,
        key=lambda record: (
            _parent_scores(record, parent_rung)[lane],
            _record_key(record),
        ),
    )
    top_half = ordered[: max(1, math.ceil(len(ordered) / 2))]
    tournament = [rng.choice(top_half) for _ in range(3)]
    parent = min(
        tournament,
        key=lambda record: (
            _parent_scores(record, parent_rung)[lane],
            _record_key(record),
        ),
    )
    return parent, lane


def _record_architecture(record):
    architecture = record["architecture"]
    return (
        architecture
        if isinstance(architecture, ArchitectureConfig)
        else ArchitectureConfig.from_dict(architecture)
    )


def later_generation(template, archive, settings, generation, existing_digests=()):
    if generation < 1:
        raise ValueError("later generations begin at one")
    rng = random.Random(derived_seed(settings["seed"], "generation", generation))
    seen = set(existing_digests)
    seen.update(record["digest"] for record in archive)
    plans = []
    child_count = settings["generation_size"] - settings["random_immigrants"]
    for _ in range(child_count):
        result = None
        parent = None
        lane = None
        for _ in range(settings["mutation_attempts"]):
            parent, lane = select_parent(archive, settings, rng)
            try:
                candidate = mutate_architecture(
                    _record_architecture(parent), settings, rng, attempts=1
                )
            except InapplicableMutation:
                continue
            if candidate.architecture.digest not in seen:
                result = candidate
                break
        if result is None:
            architecture = _unique_random(template, settings, rng, seen)
            plans.append(CandidatePlan(architecture, None, {"name": "random"}))
            continue
        seen.add(result.architecture.digest)
        plans.append(
            CandidatePlan(
                result.architecture,
                parent["digest"],
                {"name": result.name, "parent_lane": lane, **result.details},
            )
        )
    for _ in range(settings["random_immigrants"]):
        architecture = _unique_random(template, settings, rng, seen)
        plans.append(CandidatePlan(architecture, None, {"name": "random"}))
    return tuple(plans)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    text = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class StudyStore:
    directory: Path

    def __init__(self, directory):
        object.__setattr__(self, "directory", Path(directory))

    @property
    def search_path(self):
        return self.directory / "search.json"

    @property
    def state_path(self):
        return self.directory / "state.json"

    @property
    def candidates_path(self):
        return self.directory / "candidates"

    def candidate_path(self, candidate_id):
        return self.candidates_path / f"{int(candidate_id):06d}"

    def settings(self):
        return SearchSettings.from_dict(read_json(self.search_path))

    def state(self):
        return read_json(self.state_path)

    def write_state(self, state):
        value = _copy_json(state)
        now = time.time()
        if value.get("active_since") is not None:
            value["elapsed_seconds"] += max(0.0, now - value["active_since"])
            value["active_since"] = now
        value["updated_at"] = utc_now()
        atomic_json(self.state_path, value)
        return value

    def results(self):
        if not self.candidates_path.is_dir():
            return []
        values = []
        for directory in sorted(self.candidates_path.iterdir()):
            path = directory / "result.json"
            if directory.is_dir() and path.is_file():
                values.append(read_json(path))
        return values

    def architectures(self):
        return {
            result["candidate_id"]: ArchitectureConfig.from_dict(
                read_json(self.candidate_path(result["candidate_id"]) / "architecture.json")
            )
            for result in self.results()
        }

    def update_result(self, candidate_id, **changes):
        path = self.candidate_path(candidate_id) / "result.json"
        result = read_json(path)
        result.update(_copy_json(changes))
        result["updated_at"] = utc_now()
        atomic_json(path, result)
        return result


def open_study(
    directory,
    experiment,
    settings,
    hours=None,
    generations=None,
    provenance=None,
):
    store = StudyStore(directory)
    experiment = str(Path(experiment).resolve())
    provenance = _copy_json(provenance or {})
    if hours is not None and hours <= 0:
        raise ValueError("hour limit must be positive")
    if generations is not None and generations < 1:
        raise ValueError("generation limit must be positive")
    if store.state_path.exists() and not store.search_path.exists():
        raise RuntimeError("study state exists without immutable settings")
    if store.search_path.exists() and store.state_path.exists():
        stored_settings = read_json(store.search_path)
        if stored_settings != settings.settings():
            raise ValueError("comparison-sensitive search settings changed")
        state = store.state()
        if state["experiment"] != experiment:
            raise ValueError("study experiment changed")
        if state.get("provenance", {}) != provenance:
            raise ValueError("study comparison inputs or runtime changed")
        limits = dict(state["limits"])
        for name, value in (("hours", hours), ("generations", generations)):
            if value is None:
                continue
            if value < limits.get(name, 0):
                raise ValueError(f"cumulative {name} limit cannot decrease")
            limits[name] = value
        if limits != state["limits"]:
            state["limits"] = limits
            store.write_state(state)
        return store

    if hours is None and generations is None:
        raise ValueError("a new study requires an hour or generation limit")
    store.candidates_path.mkdir(parents=True, exist_ok=True)
    if any(store.candidates_path.iterdir()):
        raise RuntimeError("partial study initialization contains candidates")
    if store.search_path.exists():
        if read_json(store.search_path) != settings.settings():
            raise ValueError("comparison-sensitive search settings changed")
    else:
        atomic_json(store.search_path, settings.settings())
    now = utc_now()
    state = {
        "format_version": 1,
        "status": "running",
        "phase": "planning",
        "experiment": experiment,
        "provenance": provenance,
        "generation": 0,
        "next_candidate_id": 1,
        "seed": settings["seed"],
        "elapsed_seconds": 0.0,
        "active_since": None,
        "current_candidate": None,
        "limits": {
            "hours": hours or 0,
            "generations": generations or 0,
        },
        "started_at": now,
        "updated_at": now,
    }
    atomic_json(store.state_path, state)
    return store


def _parent_id(results, digest):
    if digest is None:
        return None
    matches = [result["candidate_id"] for result in results if result["digest"] == digest]
    if len(matches) != 1:
        raise RuntimeError("candidate parent digest is not unique in the study")
    return matches[0]


def materialize_generation(store, plans, generation, settings):
    state = store.state()
    results = store.results()
    if results:
        state["next_candidate_id"] = max(
            state["next_candidate_id"],
            max(int(result["candidate_id"]) for result in results) + 1,
        )
    by_digest = {result["digest"]: result for result in results}
    generation_results = []
    for plan in plans:
        if plan.architecture.digest in by_digest:
            existing = by_digest[plan.architecture.digest]
            if existing["generation"] != generation:
                raise RuntimeError("generation plan duplicates an earlier candidate")
            generation_results.append(existing)
            continue
        candidate_id = f"{state['next_candidate_id']:06d}"
        directory = store.candidate_path(candidate_id)
        directory.mkdir(parents=True, exist_ok=True)
        architecture_path = directory / "architecture.json"
        result_path = directory / "result.json"
        if architecture_path.exists():
            existing = ArchitectureConfig.from_dict(read_json(architecture_path))
            if existing.digest != plan.architecture.digest:
                raise RuntimeError("candidate id collides with another architecture")
        else:
            atomic_json(architecture_path, plan.architecture.settings())
        now = utc_now()
        result = {
            "format_version": 1,
            "candidate_id": candidate_id,
            "digest": plan.architecture.digest,
            "generation": generation,
            "parent": _parent_id(results, plan.parent_digest),
            "parent_digest": plan.parent_digest,
            "mutation": plan.mutation,
            "status": "pending",
            "rung": 0,
            "trained_tokens": 0,
            "nll_curve": [],
            "forecast": None,
            "profile": {"static": architecture_metrics(plan.architecture, settings)},
            "scores": {},
            "scores_by_rung": {},
            "error": None,
            "created_at": now,
            "updated_at": now,
        }
        atomic_json(result_path, result)
        results.append(result)
        by_digest[result["digest"]] = result
        generation_results.append(result)
        state["next_candidate_id"] += 1
        store.write_state(state)
    state["generation"] = generation
    state["phase"] = "screen"
    store.write_state(state)
    return tuple(generation_results)


def first_incomplete_candidate(results):
    incomplete = [
        result
        for result in results
        if result.get("status") in {"pending", "running"}
    ]
    return min(incomplete, key=lambda result: result["candidate_id"]) if incomplete else None


def prune_checkpoints(directory, keep_steps=()):
    directory = Path(directory)
    if not directory.is_dir():
        return 0
    keep = {int(step) for step in keep_steps}
    entries = []
    for path in directory.iterdir():
        match = re.fullmatch(
            r"(?:model|optimizer|metadata|complete)_(\d+)(?:\.pt|\.json)?",
            path.name,
        )
        if match and int(match.group(1)) not in keep:
            entries.append(path)
        elif path.name.endswith(".tmp"):
            entries.append(path)
    removed = sum(path.stat().st_size for path in entries if path.is_file())
    for path in entries:
        path.unlink(missing_ok=True)
    return removed


def checkpoint_disk_usage(store):
    total = 0
    if not store.candidates_path.is_dir():
        return total
    for path in store.candidates_path.glob("*/checkpoint/*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def retained_checkpoint_candidates(records, per_lane=2):
    retained = set()
    for lane in ("quality", "balanced", "efficiency"):
        eligible = sorted(
            _eligible_records(records, lane),
            key=lambda record: (record["scores"][lane], _record_key(record)),
        )
        retained.update(record["candidate_id"] for record in eligible[:per_lane])
    return retained


def validation_slices(settings):
    evaluation = settings["evaluation"]
    monitor = {
        "offset": evaluation["monitor_offset"],
        "tokens": evaluation["monitor_tokens"],
    }
    final = {
        "offset": evaluation["final_offset"],
        "tokens": evaluation["final_tokens"],
    }
    if monitor["offset"] + monitor["tokens"] > final["offset"]:
        raise ValueError("monitor and final evaluation slices overlap")
    return {"monitor": monitor, "final": final}


def loader_state(manifest, offset, sequence_length, batch_size, world_size=1):
    if offset < 0 or offset % (sequence_length * batch_size * world_size):
        raise ValueError("evaluation offset must align with loader batches")
    return {
        "format_version": 1,
        "manifest": manifest,
        "global_offset": offset,
        "epoch": 0,
        "shard": 0,
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "world_size": world_size,
    }


def status_snapshot(store):
    state = store.state()
    results = store.results()
    elapsed_seconds = state["elapsed_seconds"]
    if state.get("active_since") is not None:
        elapsed_seconds += max(0.0, time.time() - state["active_since"])
    statuses = {}
    rungs = {}
    for result in results:
        statuses[result["status"]] = statuses.get(result["status"], 0) + 1
        key = str(result.get("rung", 0))
        rungs[key] = rungs.get(key, 0) + 1
    comparable = [result for result in results if result.get("status") == "confirmed"]
    if not comparable:
        eligible = [
            result
            for result in results
            if result.get("status") != "failed" and result.get("scores")
        ]
        if eligible:
            rung = max(result.get("rung", 0) for result in eligible)
            comparable = [result for result in eligible if result.get("rung", 0) == rung]
    leaders = {
        lane: {
            "candidate_id": record["candidate_id"],
            "score": record["scores"][lane],
        }
        for lane, record in lane_leaders(comparable).items()
    }
    current = None
    if state.get("current_candidate") is not None:
        current = next(
            (
                result
                for result in results
                if result["candidate_id"] == state["current_candidate"]
            ),
            None,
        )
    return {
        "format_version": 1,
        "status": state["status"],
        "phase": state["phase"],
        "elapsed_seconds": elapsed_seconds,
        "generation": state["generation"],
        "current_candidate": current,
        "counts": {"status": dict(sorted(statuses.items())), "rung": dict(sorted(rungs.items()))},
        "leaders": leaders,
        "checkpoint_bytes": checkpoint_disk_usage(store),
    }


def select_finalists(records):
    confirmed = [record for record in records if record.get("status") == "confirmed"]
    leaders = lane_leaders(confirmed)
    missing = [
        lane
        for lane in ("quality", "balanced", "efficiency")
        if lane not in leaders
    ]
    if missing:
        raise RuntimeError(f"confirmed archive has no {', '.join(missing)} finalist")
    return {
        lane: leaders[lane]["candidate_id"]
        for lane in ("quality", "balanced", "efficiency")
    }


def aggregate_final_runs(runs, final_tokens):
    if set(runs) != {"continuation", "independent"}:
        raise ValueError("final aggregation requires continuation and independent runs")
    monitor = {}
    final = {}
    for name, run in runs.items():
        point = next(
            (
                point
                for point in run.get("nll_curve", [])
                if point["tokens"] == final_tokens
            ),
            None,
        )
        if point is None or not math.isfinite(point["nll"]):
            raise ValueError(f"{name} run has no finite final monitor nll")
        if not math.isfinite(run.get("final_nll", math.nan)):
            raise ValueError(f"{name} run has no finite untouched final nll")
        monitor[name] = point["nll"]
        final[name] = run["final_nll"]
    return {
        "runs": _copy_json(runs),
        "mean_monitor_nll": statistics.mean(monitor.values()),
        "mean_final_nll": statistics.mean(final.values()),
    }
