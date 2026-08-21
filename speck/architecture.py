"""versioned block grammar for speck model architectures."""

import json
from dataclasses import asdict, dataclass, field, replace

from speck.search.protocol import architecture_schema_version, canonical_json, content_digest


@dataclass(frozen=True)
class AttentionSpec:
    head_dim: int
    num_key_value_heads: int
    scope: str = "global"
    window_size: int | None = None
    kind: str = field(init=False, default="attention")

    def __post_init__(self):
        if self.head_dim < 2 or self.head_dim % 2:
            raise ValueError("attention head dimensions must be positive and even")
        if self.num_key_value_heads < 1:
            raise ValueError("attention kv heads must be positive")
        if self.scope not in {"global", "sliding"}:
            raise ValueError("attention scope must be global or sliding")
        if self.scope == "sliding":
            if self.window_size is None or self.window_size < 1:
                raise ValueError("sliding attention requires a positive window")
        elif self.window_size is not None:
            raise ValueError("global attention cannot define a window")


@dataclass(frozen=True)
class GatedCausalConvSpec:
    inner_size: int
    kernel_size: int
    kind: str = field(init=False, default="gated_causal_conv")

    def __post_init__(self):
        if self.inner_size < 1:
            raise ValueError("convolution inner sizes must be positive")
        if self.kernel_size < 2:
            raise ValueError("convolution kernels must contain at least two positions")


@dataclass(frozen=True)
class SwiGLUSpec:
    intermediate_size: int
    kind: str = field(init=False, default="swiglu")

    def __post_init__(self):
        if self.intermediate_size < 1:
            raise ValueError("swiglu intermediate sizes must be positive")


OperationSpec = AttentionSpec | GatedCausalConvSpec | SwiGLUSpec


def operation_from_dict(value):
    value = dict(value)
    kind = value.pop("kind")
    classes = {
        "attention": AttentionSpec,
        "gated_causal_conv": GatedCausalConvSpec,
        "swiglu": SwiGLUSpec,
    }
    if kind not in classes:
        raise ValueError(f"unknown architecture operation: {kind}")
    return classes[kind](**value)


@dataclass(frozen=True)
class StageConfig:
    branches: tuple[OperationSpec, ...]

    def __post_init__(self):
        if not self.branches:
            raise ValueError("architecture stages cannot be empty")
        kinds = tuple(branch.kind for branch in self.branches)
        if len(set(kinds)) != len(kinds):
            raise ValueError("parallel stage operation kinds must be unique")

    @classmethod
    def from_dict(cls, value):
        return cls(tuple(operation_from_dict(item) for item in value["branches"]))


@dataclass(frozen=True)
class BlockConfig:
    hidden_size: int
    stages: tuple[StageConfig, ...]

    def __post_init__(self):
        if self.hidden_size < 1:
            raise ValueError("block hidden sizes must be positive")
        if not self.stages:
            raise ValueError("architecture blocks cannot be empty")
        for stage in self.stages:
            for operation in stage.branches:
                if isinstance(operation, AttentionSpec):
                    if self.hidden_size % operation.head_dim:
                        raise ValueError("block width must be divisible by attention head dimension")
                    query_heads = self.hidden_size // operation.head_dim
                    if query_heads % operation.num_key_value_heads:
                        raise ValueError("query heads must be divisible by kv heads")

    @classmethod
    def from_dict(cls, value):
        return cls(
            hidden_size=value["hidden_size"],
            stages=tuple(StageConfig.from_dict(stage) for stage in value["stages"]),
        )


@dataclass(frozen=True)
class BlockGroup:
    block: BlockConfig
    repeat: int = 1
    weight_sharing: str = "none"

    def __post_init__(self):
        if self.repeat < 1:
            raise ValueError("block repeat counts must be positive")
        if self.weight_sharing not in {"none", "all"}:
            raise ValueError("weight sharing must be none or all")

    @classmethod
    def from_dict(cls, value):
        return cls(
            block=BlockConfig.from_dict(value["block"]),
            repeat=value.get("repeat", 1),
            weight_sharing=value.get("weight_sharing", "none"),
        )


@dataclass(frozen=True)
class BlockInvocation:
    block: BlockConfig
    occurrence_index: int
    group_index: int
    repeat_index: int
    weight_key: str


@dataclass(frozen=True)
class ArchitectureConfig:
    blocks: tuple[BlockGroup, ...]
    embedding_size: int
    default_head_dim: int = 64
    vocab_size: int = 32_000
    bos_token_id: int = 1
    eos_token_id: int = 2
    max_position_embeddings: int = 4_096
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    expected_parameters: int | None = None
    architecture: str = "speck"
    architecture_version: int = architecture_schema_version

    def __post_init__(self):
        if self.architecture != "speck":
            raise ValueError("architecture must be speck")
        if self.architecture_version != architecture_schema_version:
            raise ValueError("unsupported architecture version")
        if not self.blocks:
            raise ValueError("architectures need at least one block")
        if self.embedding_size < 1 or self.vocab_size < 1:
            raise ValueError("embedding and vocabulary sizes must be positive")
        if self.default_head_dim < 2 or self.default_head_dim % 2:
            raise ValueError("default head dimensions must be positive and even")
        if self.max_position_embeddings < 1:
            raise ValueError("maximum positions must be positive")
        if self.rope_theta <= 0 or self.rms_norm_eps <= 0 or self.initializer_range <= 0:
            raise ValueError("model scaling values must be positive")
        if not 0 <= self.bos_token_id < self.vocab_size:
            raise ValueError("bos token id is outside the vocabulary")
        if not 0 <= self.eos_token_id < self.vocab_size:
            raise ValueError("eos token id is outside the vocabulary")
        if self.expected_parameters is not None and self.expected_parameters < 1:
            raise ValueError("expected parameters must be positive")

    @property
    def logical_depth(self):
        return sum(group.repeat for group in self.blocks)

    @property
    def execution_plan(self):
        invocations = []
        occurrence = 0
        for group_index, group in enumerate(self.blocks):
            for repeat_index in range(group.repeat):
                weight_repeat = 0 if group.weight_sharing == "all" else repeat_index
                invocations.append(
                    BlockInvocation(
                        block=group.block,
                        occurrence_index=occurrence,
                        group_index=group_index,
                        repeat_index=repeat_index,
                        weight_key=f"group_{group_index}_repeat_{weight_repeat}",
                    )
                )
                occurrence += 1
        return tuple(invocations)

    @property
    def unique_parameter_blocks(self):
        return len({invocation.weight_key for invocation in self.execution_plan})

    def settings(self):
        groups = []
        for group in self.blocks:
            normalized = replace(
                group,
                weight_sharing=(
                    "none" if group.repeat == 1 else group.weight_sharing
                ),
            )
            if normalized.weight_sharing == "none" and normalized.repeat > 1:
                groups.extend(replace(normalized, repeat=1) for _ in range(normalized.repeat))
            else:
                groups.append(normalized)
        values = asdict(replace(self, blocks=tuple(groups)))
        values.pop("expected_parameters")
        return json.loads(canonical_json(values))

    def export(self):
        values = self.settings()
        values.update(
            hidden_size=self.embedding_size,
            num_hidden_layers=self.logical_depth,
            tie_word_embeddings=True,
        )
        if self.expected_parameters is not None:
            values["expected_parameters"] = self.expected_parameters
        return values

    @property
    def digest(self):
        return content_digest(self.settings())

    @classmethod
    def from_v2(cls, config):
        blocks = []
        for layer in config.layers:
            stages = []
            if layer.num_key_value_heads is not None:
                stages.append(
                    StageConfig((AttentionSpec(config.head_dim, layer.num_key_value_heads),))
                )
            stages.append(StageConfig((SwiGLUSpec(layer.intermediate_size),)))
            blocks.append(BlockGroup(BlockConfig(layer.hidden_size, tuple(stages))))
        return cls(
            architecture=getattr(config, "architecture", "speck"),
            blocks=tuple(blocks),
            embedding_size=config.embedding_size,
            default_head_dim=config.head_dim,
            vocab_size=config.vocab_size,
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            initializer_range=config.initializer_range,
            expected_parameters=getattr(config, "expected_parameters", None),
        )

    def to_v2(self):
        from speck.model import Config, LayerConfig

        if any(
            group.weight_sharing == "all" and group.repeat > 1
            for group in self.blocks
        ):
            raise ValueError("shared blocks cannot be represented by v2")
        head_dimensions = set()
        layers = []
        for invocation in self.execution_plan:
            stages = invocation.block.stages
            if any(len(stage.branches) != 1 for stage in stages):
                raise ValueError("parallel stages cannot be represented by v2")
            operations = tuple(stage.branches[0] for stage in stages)
            attention = tuple(
                operation for operation in operations if isinstance(operation, AttentionSpec)
            )
            convolutions = tuple(
                operation
                for operation in operations
                if isinstance(operation, GatedCausalConvSpec)
            )
            mlps = tuple(
                operation for operation in operations if isinstance(operation, SwiGLUSpec)
            )
            expected_order = attention + mlps
            if convolutions or len(attention) > 1 or len(mlps) != 1 or operations != expected_order:
                raise ValueError("v3 block cannot be represented by v2")
            if attention and attention[0].scope != "global":
                raise ValueError("local attention cannot be represented by v2")
            if attention:
                head_dimensions.add(attention[0].head_dim)
            layers.append(
                LayerConfig(
                    invocation.block.hidden_size,
                    mlps[0].intermediate_size,
                    attention[0].num_key_value_heads if attention else None,
                )
            )
        if len(head_dimensions) > 1:
            raise ValueError("heterogeneous head dimensions cannot be represented by v2")
        head_dim = next(iter(head_dimensions), self.default_head_dim)
        return Config(
            vocab_size=self.vocab_size,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            layers=tuple(layers),
            head_dim=head_dim,
            max_position_embeddings=self.max_position_embeddings,
            rope_theta=self.rope_theta,
            rms_norm_eps=self.rms_norm_eps,
            initializer_range=self.initializer_range,
        )

    @classmethod
    def from_dict(cls, value):
        if value.get("architecture_version") != architecture_schema_version:
            from speck.model import Config

            converted = cls.from_v2(Config.from_dict(value))
            return replace(
                converted,
                architecture=value.get("architecture", "speck"),
                expected_parameters=value.get("expected_parameters"),
            )
        values = dict(value)
        for key in (
            "hidden_size",
            "num_hidden_layers",
            "tie_word_embeddings",
        ):
            values.pop(key, None)
        values["blocks"] = tuple(BlockGroup.from_dict(block) for block in values["blocks"])
        return cls(**values)
