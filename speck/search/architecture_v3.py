"""search operators and exact accounting for version three architectures."""

import random
from dataclasses import dataclass, replace

import torch

from speck.architecture import (
    ArchitectureConfig,
    AttentionSpec,
    BlockConfig,
    BlockGroup,
    GatedCausalConvSpec,
    StageConfig,
    SwiGLUSpec,
)
from speck.model_v3 import SpeckV3ForCausalLM


mutation_operators = (
    "add_block",
    "remove_block",
    "change_hidden_size",
    "change_mixer",
    "change_attention_scope",
    "change_head_dim",
    "change_kv_heads",
    "change_sliding_window",
    "change_conv_kernel",
    "change_conv_width",
    "toggle_swiglu",
    "change_swiglu_width",
    "change_repeat",
    "toggle_weight_sharing",
)


def _choices(values, name):
    values = tuple(sorted(set(values)))
    if not values or values[0] < 1:
        raise ValueError(f"{name} choices must be positive")
    return values


@dataclass(frozen=True)
class V3SearchSpace:
    min_logical_depth: int
    max_logical_depth: int
    hidden_sizes: tuple[int, ...]
    intermediate_sizes: tuple[int, ...]
    head_dims: tuple[int, ...]
    kv_heads: tuple[int, ...]
    sliding_windows: tuple[int, ...]
    conv_kernel_sizes: tuple[int, ...]
    conv_inner_sizes: tuple[int, ...]
    repeat_counts: tuple[int, ...] = (1, 2)
    mixer_kinds: tuple[str, ...] = ("none", "attention", "gated_causal_conv")
    attention_scopes: tuple[str, ...] = ("global", "sliding")
    allow_swiglu: bool = True
    allow_mixer_only: bool = True

    def __post_init__(self):
        if self.min_logical_depth < 1 or self.max_logical_depth < self.min_logical_depth:
            raise ValueError("invalid logical depth range")
        for name in (
            "hidden_sizes",
            "intermediate_sizes",
            "head_dims",
            "kv_heads",
            "sliding_windows",
            "conv_kernel_sizes",
            "conv_inner_sizes",
            "repeat_counts",
        ):
            object.__setattr__(self, name, _choices(getattr(self, name), name))
        if any(head_dim % 2 for head_dim in self.head_dims):
            raise ValueError("attention head dimensions must be even")
        valid_mixers = {"none", "attention", "gated_causal_conv"}
        if not self.mixer_kinds or set(self.mixer_kinds) - valid_mixers:
            raise ValueError("invalid mixer choices")
        if not self.attention_scopes or set(self.attention_scopes) - {"global", "sliding"}:
            raise ValueError("invalid attention scope choices")
        if not self.allow_swiglu and "none" in self.mixer_kinds:
            raise ValueError("empty blocks cannot be part of the search space")
        if not self.allow_swiglu and not self.allow_mixer_only:
            raise ValueError("search space cannot construct a block")
        if "attention" in self.mixer_kinds and any(
            not _attention_choices(hidden_size, self)
            for hidden_size in self.hidden_sizes
        ):
            raise ValueError("every hidden size needs a valid attention configuration")

    @classmethod
    def from_dict(cls, value):
        values = dict(value)
        for name in (
            "hidden_sizes",
            "intermediate_sizes",
            "head_dims",
            "kv_heads",
            "sliding_windows",
            "conv_kernel_sizes",
            "conv_inner_sizes",
            "repeat_counts",
            "mixer_kinds",
            "attention_scopes",
        ):
            if name in values:
                values[name] = tuple(values[name])
        return cls(**values)


@dataclass(frozen=True)
class MutationResult:
    config: ArchitectureConfig
    operation: dict


def architecture_hash(config):
    return config.digest


def _block_operations(block):
    if any(len(stage.branches) != 1 for stage in block.stages):
        raise ValueError("v3 search requires singleton stages")
    operations = tuple(stage.branches[0] for stage in block.stages)
    mixers = tuple(
        operation
        for operation in operations
        if isinstance(operation, (AttentionSpec, GatedCausalConvSpec))
    )
    mlps = tuple(operation for operation in operations if isinstance(operation, SwiGLUSpec))
    if len(mixers) > 1 or len(mlps) > 1 or operations != mixers + mlps:
        raise ValueError("v3 search requires an optional mixer followed by optional swiglu")
    return (mixers[0] if mixers else None), (mlps[0] if mlps else None)


def _make_block(hidden_size, mixer, mlp):
    operations = tuple(operation for operation in (mixer, mlp) if operation is not None)
    if not operations:
        raise ValueError("search blocks cannot be empty")
    return BlockConfig(
        hidden_size,
        tuple(StageConfig((operation,)) for operation in operations),
    )


def _nearest(value, choices):
    return min(choices, key=lambda choice: (abs(choice - value), choice))


def _neighbor(value, choices, rng):
    index = choices.index(value)
    neighbors = []
    if index:
        neighbors.append(choices[index - 1])
    if index + 1 < len(choices):
        neighbors.append(choices[index + 1])
    if not neighbors:
        raise ValueError("value has no neighboring choice")
    return rng.choice(neighbors)


def _attention_choices(hidden_size, space):
    values = []
    for head_dim in space.head_dims:
        if hidden_size % head_dim:
            continue
        query_heads = hidden_size // head_dim
        for kv_heads in space.kv_heads:
            if query_heads % kv_heads == 0:
                values.append((head_dim, kv_heads))
    return tuple(values)


def _repair_attention(hidden_size, attention, space):
    choices = _attention_choices(hidden_size, space)
    if not choices:
        raise ValueError("hidden size has no valid attention configuration")
    head_dim, kv_heads = min(
        choices,
        key=lambda value: (
            abs(value[0] - attention.head_dim),
            abs(value[1] - attention.num_key_value_heads),
            value,
        ),
    )
    scope = attention.scope if attention.scope in space.attention_scopes else space.attention_scopes[0]
    window = (
        _nearest(attention.window_size or space.sliding_windows[0], space.sliding_windows)
        if scope == "sliding"
        else None
    )
    return AttentionSpec(head_dim, kv_heads, scope, window)


def repair(config, space):
    groups = []
    for group in config.blocks:
        mixer, mlp = _block_operations(group.block)
        hidden_size = _nearest(group.block.hidden_size, space.hidden_sizes)
        if isinstance(mixer, AttentionSpec):
            mixer = _repair_attention(hidden_size, mixer, space)
        elif isinstance(mixer, GatedCausalConvSpec):
            mixer = GatedCausalConvSpec(
                _nearest(mixer.inner_size, space.conv_inner_sizes),
                _nearest(mixer.kernel_size, space.conv_kernel_sizes),
            )
        if mlp is not None:
            if not space.allow_swiglu:
                mlp = None
            else:
                mlp = SwiGLUSpec(
                    _nearest(mlp.intermediate_size, space.intermediate_sizes)
                )
        elif not space.allow_mixer_only:
            mlp = SwiGLUSpec(space.intermediate_sizes[0])
        if mixer is None and mlp is None:
            if space.allow_swiglu:
                mlp = SwiGLUSpec(space.intermediate_sizes[0])
            else:
                mixer = _random_mixer(hidden_size, space, random.Random(0), include_none=False)
        repeat = _nearest(group.repeat, space.repeat_counts)
        sharing = group.weight_sharing if repeat > 1 else "none"
        groups.append(BlockGroup(_make_block(hidden_size, mixer, mlp), repeat, sharing))

    depth = sum(group.repeat for group in groups)
    while depth < space.min_logical_depth:
        source = groups[-1]
        groups.append(replace(source, repeat=1, weight_sharing="none"))
        depth += 1
    while depth > space.max_logical_depth:
        group = groups[-1]
        excess = depth - space.max_logical_depth
        if group.repeat > excess:
            groups[-1] = replace(group, repeat=group.repeat - excess)
            depth -= excess
        else:
            depth -= group.repeat
            groups.pop()
    return replace(config, blocks=tuple(groups), expected_parameters=None)


def parameter_count(config):
    total = config.vocab_size * config.embedding_size
    input_size = config.embedding_size
    counted = set()
    for invocation in config.execution_plan:
        hidden_size = invocation.block.hidden_size
        if input_size != hidden_size:
            total += input_size * hidden_size
        if invocation.weight_key not in counted:
            counted.add(invocation.weight_key)
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
        input_size = hidden_size
    total += input_size
    if input_size != config.embedding_size:
        total += input_size * config.embedding_size
    return total


def state_bytes(config, context, batch_size=1, dtype_bytes=2):
    if context < 1 or batch_size < 1 or dtype_bytes < 1:
        raise ValueError("state accounting values must be positive")
    total = 0
    for invocation in config.execution_plan:
        for stage in invocation.block.stages:
            for operation in stage.branches:
                if isinstance(operation, AttentionSpec):
                    positions = context if operation.scope == "global" else min(
                        context, operation.window_size
                    )
                    total += (
                        2
                        * batch_size
                        * positions
                        * operation.num_key_value_heads
                        * operation.head_dim
                        * dtype_bytes
                    )
                elif isinstance(operation, GatedCausalConvSpec):
                    total += (
                        batch_size
                        * operation.inner_size
                        * (operation.kernel_size - 1)
                        * dtype_bytes
                    )
    return total


def quantized_weight_bytes(config, bits=4, group_size=128, scale_bytes=2, other_bytes=2):
    if not 1 <= bits <= 8 or min(group_size, scale_bytes, other_bytes) < 1:
        raise ValueError("invalid quantized weight accounting settings")
    with torch.device("meta"):
        model = SpeckV3ForCausalLM(config)
    total = 0
    for parameter in model.parameters():
        if parameter.ndim == 2:
            rows, columns = parameter.shape
            packed = rows * ((columns * bits + 7) // 8)
            scales = rows * ((columns + group_size - 1) // group_size) * scale_bytes
            total += packed + scales
        else:
            total += parameter.numel() * other_bytes
    return total


def _random_attention(hidden_size, space, rng):
    choices = _attention_choices(hidden_size, space)
    if not choices:
        raise ValueError("hidden size has no valid attention configuration")
    head_dim, kv_heads = rng.choice(choices)
    scope = rng.choice(space.attention_scopes)
    window = rng.choice(space.sliding_windows) if scope == "sliding" else None
    return AttentionSpec(head_dim, kv_heads, scope, window)


def _random_mixer(hidden_size, space, rng, include_none=True):
    kinds = tuple(
        kind for kind in space.mixer_kinds if include_none or kind != "none"
    )
    kind = rng.choice(kinds)
    if kind == "none":
        return None
    if kind == "attention":
        return _random_attention(hidden_size, space, rng)
    return GatedCausalConvSpec(
        rng.choice(space.conv_inner_sizes),
        rng.choice(space.conv_kernel_sizes),
    )


def sample_architecture(base, space, seed):
    rng = random.Random(seed)
    depth = rng.randint(space.min_logical_depth, space.max_logical_depth)
    groups = []
    while sum(group.repeat for group in groups) < depth:
        remaining = depth - sum(group.repeat for group in groups)
        repeat = rng.choice(tuple(value for value in space.repeat_counts if value <= remaining))
        hidden_size = rng.choice(space.hidden_sizes)
        mixer = _random_mixer(hidden_size, space, rng)
        mlp = (
            SwiGLUSpec(rng.choice(space.intermediate_sizes))
            if space.allow_swiglu
            and (
                mixer is None
                or not space.allow_mixer_only
                or rng.choice((False, True))
            )
            else None
        )
        if mixer is None and mlp is None:
            mlp = SwiGLUSpec(rng.choice(space.intermediate_sizes))
        sharing = rng.choice(("none", "all")) if repeat > 1 else "none"
        groups.append(BlockGroup(_make_block(hidden_size, mixer, mlp), repeat, sharing))
    return replace(base, blocks=tuple(groups), expected_parameters=None)


def available_mutations(config, space):
    available = []
    depth = config.logical_depth
    if depth < space.max_logical_depth:
        available.append("add_block")
    if depth > space.min_logical_depth and len(config.blocks) > 1:
        available.append("remove_block")
    if len(space.hidden_sizes) > 1:
        available.append("change_hidden_size")
    if len(space.mixer_kinds) > 1:
        available.append("change_mixer")
    attentions = [
        (index, mixer)
        for index, group in enumerate(config.blocks)
        if isinstance((mixer := _block_operations(group.block)[0]), AttentionSpec)
    ]
    convolutions = [
        (index, mixer)
        for index, group in enumerate(config.blocks)
        if isinstance((mixer := _block_operations(group.block)[0]), GatedCausalConvSpec)
    ]
    mlps = [
        (index, mlp)
        for index, group in enumerate(config.blocks)
        if (mlp := _block_operations(group.block)[1]) is not None
    ]
    if attentions and len(space.attention_scopes) > 1:
        available.append("change_attention_scope")
    if any(
        any(
            head_dim != attention.head_dim
            and config.blocks[index].block.hidden_size % head_dim == 0
            and (config.blocks[index].block.hidden_size // head_dim)
            % attention.num_key_value_heads
            == 0
            for head_dim in space.head_dims
        )
        for index, attention in attentions
    ):
        available.append("change_head_dim")
    if any(
        any(
            kv_heads != attention.num_key_value_heads
            and (config.blocks[index].block.hidden_size // attention.head_dim)
            % kv_heads
            == 0
            for kv_heads in space.kv_heads
        )
        for index, attention in attentions
    ):
        available.append("change_kv_heads")
    if any(attention.scope == "sliding" for _, attention in attentions) and len(space.sliding_windows) > 1:
        available.append("change_sliding_window")
    if convolutions and len(space.conv_kernel_sizes) > 1:
        available.append("change_conv_kernel")
    if convolutions and len(space.conv_inner_sizes) > 1:
        available.append("change_conv_width")
    if space.allow_swiglu and any(
        mlp is None or (mixer is not None and space.allow_mixer_only)
        for group in config.blocks
        for mixer, mlp in (_block_operations(group.block),)
    ):
        available.append("toggle_swiglu")
    if mlps and len(space.intermediate_sizes) > 1:
        available.append("change_swiglu_width")
    if any(
        any(
            repeat != group.repeat
            and space.min_logical_depth
            <= config.logical_depth - group.repeat + repeat
            <= space.max_logical_depth
            for repeat in space.repeat_counts
        )
        for group in config.blocks
    ):
        available.append("change_repeat")
    if any(group.repeat > 1 for group in config.blocks):
        available.append("toggle_weight_sharing")
    return tuple(available)


def _replace_block(group, hidden_size=None, mixer=None, mlp=None, keep_mixer=True, keep_mlp=True):
    current_mixer, current_mlp = _block_operations(group.block)
    return replace(
        group,
        block=_make_block(
            hidden_size or group.block.hidden_size,
            current_mixer if keep_mixer else mixer,
            current_mlp if keep_mlp else mlp,
        ),
    )


def mutate(config, space, seed, operator=None):
    rng = random.Random(seed)
    available = available_mutations(config, space)
    operator = operator or rng.choice(available)
    if operator not in available:
        raise ValueError(f"mutation is not available: {operator}")
    groups = list(config.blocks)
    operation = {"operator": operator, "seed": seed}

    if operator == "add_block":
        index = rng.randrange(len(groups) + 1)
        source = groups[min(index, len(groups) - 1)]
        groups.insert(index, replace(source, repeat=1, weight_sharing="none"))
        operation.update(index=index)
    elif operator == "remove_block":
        index = rng.randrange(len(groups))
        removed = groups.pop(index)
        operation.update(index=index, removed=removed.block.hidden_size)
    elif operator == "change_hidden_size":
        index = rng.randrange(len(groups))
        group = groups[index]
        hidden_size = _neighbor(group.block.hidden_size, space.hidden_sizes, rng)
        mixer, mlp = _block_operations(group.block)
        if isinstance(mixer, AttentionSpec):
            mixer = _repair_attention(hidden_size, mixer, space)
        groups[index] = replace(group, block=_make_block(hidden_size, mixer, mlp))
        operation.update(index=index, hidden_size=hidden_size)
    elif operator == "change_mixer":
        index = rng.randrange(len(groups))
        group = groups[index]
        current, mlp = _block_operations(group.block)
        current_kind = current.kind if current is not None else "none"
        kinds = tuple(kind for kind in space.mixer_kinds if kind != current_kind)
        if mlp is None:
            kinds = tuple(kind for kind in kinds if kind != "none")
        kind = rng.choice(kinds)
        if kind == "none":
            mixer = None
        elif kind == "attention":
            mixer = _random_attention(group.block.hidden_size, space, rng)
        else:
            mixer = GatedCausalConvSpec(
                rng.choice(space.conv_inner_sizes),
                rng.choice(space.conv_kernel_sizes),
            )
        groups[index] = replace(group, block=_make_block(group.block.hidden_size, mixer, mlp))
        operation.update(index=index, mixer=kind)
    elif operator in {
        "change_attention_scope",
        "change_head_dim",
        "change_kv_heads",
        "change_sliding_window",
    }:
        indices = [
            index
            for index, group in enumerate(groups)
            if isinstance(_block_operations(group.block)[0], AttentionSpec)
        ]
        if operator == "change_sliding_window":
            indices = [
                index
                for index in indices
                if _block_operations(groups[index].block)[0].scope == "sliding"
            ]
        elif operator == "change_head_dim":
            indices = [
                index
                for index in indices
                if any(
                    head_dim != _block_operations(groups[index].block)[0].head_dim
                    and groups[index].block.hidden_size % head_dim == 0
                    and (groups[index].block.hidden_size // head_dim)
                    % _block_operations(groups[index].block)[0].num_key_value_heads
                    == 0
                    for head_dim in space.head_dims
                )
            ]
        elif operator == "change_kv_heads":
            indices = [
                index
                for index in indices
                if any(
                    kv_heads
                    != _block_operations(groups[index].block)[0].num_key_value_heads
                    and (
                        groups[index].block.hidden_size
                        // _block_operations(groups[index].block)[0].head_dim
                    )
                    % kv_heads
                    == 0
                    for kv_heads in space.kv_heads
                )
            ]
        index = rng.choice(indices)
        group = groups[index]
        attention, mlp = _block_operations(group.block)
        if operator == "change_attention_scope":
            scope = rng.choice(tuple(value for value in space.attention_scopes if value != attention.scope))
            window = rng.choice(space.sliding_windows) if scope == "sliding" else None
            attention = replace(attention, scope=scope, window_size=window)
        elif operator == "change_head_dim":
            choices = tuple(
                value
                for value in space.head_dims
                if value != attention.head_dim
                and group.block.hidden_size % value == 0
                and (group.block.hidden_size // value) % attention.num_key_value_heads == 0
            )
            if not choices:
                raise ValueError("attention has no alternate head dimension")
            attention = replace(attention, head_dim=rng.choice(choices))
        elif operator == "change_kv_heads":
            query_heads = group.block.hidden_size // attention.head_dim
            choices = tuple(
                value
                for value in space.kv_heads
                if value != attention.num_key_value_heads and query_heads % value == 0
            )
            if not choices:
                raise ValueError("attention has no alternate kv head count")
            attention = replace(attention, num_key_value_heads=rng.choice(choices))
        else:
            attention = replace(
                attention,
                window_size=_neighbor(attention.window_size, space.sliding_windows, rng),
            )
        groups[index] = replace(group, block=_make_block(group.block.hidden_size, attention, mlp))
        operation.update(index=index)
    elif operator in {"change_conv_kernel", "change_conv_width"}:
        indices = [
            index
            for index, group in enumerate(groups)
            if isinstance(_block_operations(group.block)[0], GatedCausalConvSpec)
        ]
        index = rng.choice(indices)
        group = groups[index]
        convolution, mlp = _block_operations(group.block)
        if operator == "change_conv_kernel":
            convolution = replace(
                convolution,
                kernel_size=_neighbor(
                    convolution.kernel_size, space.conv_kernel_sizes, rng
                ),
            )
        else:
            convolution = replace(
                convolution,
                inner_size=_neighbor(
                    convolution.inner_size, space.conv_inner_sizes, rng
                ),
            )
        groups[index] = replace(group, block=_make_block(group.block.hidden_size, convolution, mlp))
        operation.update(index=index)
    elif operator == "toggle_swiglu":
        indices = []
        for index, group in enumerate(groups):
            mixer, mlp = _block_operations(group.block)
            if mlp is not None and mixer is None:
                continue
            indices.append(index)
        index = rng.choice(indices)
        group = groups[index]
        mixer, mlp = _block_operations(group.block)
        mlp = None if mlp is not None else SwiGLUSpec(rng.choice(space.intermediate_sizes))
        groups[index] = replace(group, block=_make_block(group.block.hidden_size, mixer, mlp))
        operation.update(index=index, enabled=mlp is not None)
    elif operator == "change_swiglu_width":
        indices = [
            index
            for index, group in enumerate(groups)
            if _block_operations(group.block)[1] is not None
        ]
        index = rng.choice(indices)
        group = groups[index]
        mixer, mlp = _block_operations(group.block)
        mlp = SwiGLUSpec(
            _neighbor(mlp.intermediate_size, space.intermediate_sizes, rng)
        )
        groups[index] = replace(group, block=_make_block(group.block.hidden_size, mixer, mlp))
        operation.update(index=index, intermediate_size=mlp.intermediate_size)
    elif operator == "change_repeat":
        indices = [
            index
            for index, group in enumerate(groups)
            if any(
                repeat != group.repeat
                and space.min_logical_depth
                <= config.logical_depth - group.repeat + repeat
                <= space.max_logical_depth
                for repeat in space.repeat_counts
            )
        ]
        index = rng.choice(indices)
        group = groups[index]
        choices = tuple(
            value
            for value in space.repeat_counts
            if value != group.repeat
            and config.logical_depth - group.repeat + value >= space.min_logical_depth
            and config.logical_depth - group.repeat + value <= space.max_logical_depth
        )
        if not choices:
            raise ValueError("block group has no valid repeat change")
        repeat = rng.choice(choices)
        groups[index] = replace(
            group,
            repeat=repeat,
            weight_sharing=group.weight_sharing if repeat > 1 else "none",
        )
        operation.update(index=index, repeat=repeat)
    else:
        indices = [index for index, group in enumerate(groups) if group.repeat > 1]
        index = rng.choice(indices)
        group = groups[index]
        sharing = "none" if group.weight_sharing == "all" else "all"
        groups[index] = replace(group, weight_sharing=sharing)
        operation.update(index=index, weight_sharing=sharing)

    mutated = repair(replace(config, blocks=tuple(groups), expected_parameters=None), space)
    if mutated.digest == config.digest:
        raise ValueError("mutation did not change the architecture")
    return MutationResult(mutated, operation)


def crossover(left, right, space, seed):
    if len(left.blocks) < 2 or len(right.blocks) < 2:
        raise ValueError("crossover parents need at least two block groups")
    left_global = left.settings()
    right_global = right.settings()
    left_global.pop("blocks")
    right_global.pop("blocks")
    if left_global != right_global:
        raise ValueError("crossover parents must share global model settings")
    rng = random.Random(seed)
    left_cut = rng.randrange(1, len(left.blocks))
    right_cut = rng.randrange(1, len(right.blocks))
    child = repair(
        replace(
            left,
            blocks=left.blocks[:left_cut] + right.blocks[right_cut:],
            expected_parameters=None,
        ),
        space,
    )
    return MutationResult(
        child,
        {
            "operator": "crossover",
            "seed": seed,
            "left_hash": left.digest,
            "right_hash": right.digest,
            "left_cut": left_cut,
            "right_cut": right_cut,
        },
    )


def architecture_distance(left, right, space):
    left_plan = left.execution_plan
    right_plan = right.execution_plan
    maximum = max(len(left_plan), len(right_plan), 1)
    hidden_span = max(space.hidden_sizes) - min(space.hidden_sizes) or 1
    distance = 0.0
    for index in range(maximum):
        if index >= len(left_plan) or index >= len(right_plan):
            distance += 1.0
            continue
        left_block = left_plan[index].block
        right_block = right_plan[index].block
        distance += abs(left_block.hidden_size - right_block.hidden_size) / hidden_span
        left_mixer, left_mlp = _block_operations(left_block)
        right_mixer, right_mlp = _block_operations(right_block)
        left_kind = left_mixer.kind if left_mixer is not None else "none"
        right_kind = right_mixer.kind if right_mixer is not None else "none"
        distance += float(left_kind != right_kind)
        distance += float((left_mlp is None) != (right_mlp is None))
        distance += float(left_plan[index].weight_key != right_plan[index].weight_key)
    return distance / (4 * maximum)
