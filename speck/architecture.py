"""Define the block grammar for Speck model architectures."""

import json
from dataclasses import asdict, dataclass, field, replace


def canonical_json(value):
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


@dataclass(frozen=True)
class AttentionSpec:
    head_dim: int
    num_key_value_heads: int
    scope: str = "global"
    window_size: int | None = None
    rope_dim: int | None = None
    kind: str = field(init=False, default="attention")

    def __post_init__(self):
        if self.head_dim < 2 or self.head_dim % 2:
            raise ValueError("attention head dimensions must be positive and even")
        if self.num_key_value_heads < 1:
            raise ValueError("attention KV heads must be positive")
        if self.scope not in {"global", "sliding"}:
            raise ValueError("attention scope must be global or sliding")
        if self.scope == "sliding":
            if self.window_size is None or self.window_size < 1:
                raise ValueError("sliding attention requires a positive window")
        elif self.window_size is not None:
            raise ValueError("global attention cannot define a window")
        rope_dim = self.head_dim if self.rope_dim is None else self.rope_dim
        if rope_dim < 0 or rope_dim > self.head_dim or rope_dim % 2:
            raise ValueError("attention RoPE dimensions must be even and within the head")


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
class GatedDeltaNetSpec:
    key_head_dim: int
    value_head_dim: int
    num_key_heads: int
    num_value_heads: int
    conv_kernel_size: int = 4
    output_gate_activation: str = "silu"
    kind: str = field(init=False, default="gated_deltanet")

    def __post_init__(self):
        dimensions = (
            self.key_head_dim,
            self.value_head_dim,
            self.num_key_heads,
            self.num_value_heads,
        )
        if any(value < 1 for value in dimensions):
            raise ValueError("Gated DeltaNet dimensions and head counts must be positive")
        if self.num_value_heads % self.num_key_heads:
            raise ValueError("Gated DeltaNet value heads must be divisible by key heads")
        if self.conv_kernel_size < 2:
            raise ValueError("Gated DeltaNet convolution kernels need at least two positions")
        if self.output_gate_activation not in {"sigmoid", "silu"}:
            raise ValueError("Gated DeltaNet output gate activation must be sigmoid or silu")


@dataclass(frozen=True)
class SwiGLUSpec:
    intermediate_size: int
    kind: str = field(init=False, default="swiglu")

    def __post_init__(self):
        if self.intermediate_size < 1:
            raise ValueError("SwiGLU intermediate sizes must be positive")


OperationSpec = AttentionSpec | GatedCausalConvSpec | GatedDeltaNetSpec | SwiGLUSpec


def operation_from_dict(value):
    value = dict(value)
    kind = value.pop("kind")
    classes = {
        "attention": AttentionSpec,
        "gated_causal_conv": GatedCausalConvSpec,
        "gated_deltanet": GatedDeltaNetSpec,
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
                        raise ValueError(
                            "block width must be divisible by attention head dimension"
                        )
                    query_heads = self.hidden_size // operation.head_dim
                    if query_heads % operation.num_key_value_heads:
                        raise ValueError("query heads must be divisible by KV heads")

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
    weight_key: str


@dataclass(frozen=True)
class ArchitectureConfig:
    """Describe a Speck model as an ordered sequence of optionally shared block groups."""

    blocks: tuple[BlockGroup, ...]
    embedding_size: int
    vocab_size: int = 32_000
    bos_token_id: int = 1
    eos_token_id: int = 2
    max_position_embeddings: int = 4_096
    rope_theta: float = 10_000.0
    rope_scaling_factor: float = 1.0
    rms_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    expected_parameters: int | None = None

    def __post_init__(self):
        if not self.blocks:
            raise ValueError("architectures need at least one block")
        if self.embedding_size < 1 or self.vocab_size < 1:
            raise ValueError("embedding and vocabulary sizes must be positive")
        if self.max_position_embeddings < 1:
            raise ValueError("maximum positions must be positive")
        if (
            self.rope_theta <= 0
            or self.rope_scaling_factor < 1
            or self.rms_norm_eps <= 0
            or self.initializer_range <= 0
        ):
            raise ValueError("model scaling values must be positive")
        if not 0 <= self.bos_token_id < self.vocab_size:
            raise ValueError("BOS token ID is outside the vocabulary")
        if not 0 <= self.eos_token_id < self.vocab_size:
            raise ValueError("EOS token ID is outside the vocabulary")
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
                        weight_key=f"group_{group_index}_repeat_{weight_repeat}",
                    )
                )
                occurrence += 1
        return tuple(invocations)

    def settings(self):
        groups = tuple(
            replace(
                group,
                weight_sharing="none" if group.repeat == 1 else group.weight_sharing,
            )
            for group in self.blocks
        )
        values = asdict(replace(self, blocks=groups))
        values.pop("expected_parameters")
        return json.loads(canonical_json(values))

    def export(self):
        values = self.settings()
        if self.expected_parameters is not None:
            values["expected_parameters"] = self.expected_parameters
        return values

    @classmethod
    def from_dict(cls, value):
        values = dict(value)
        values["blocks"] = tuple(BlockGroup.from_dict(block) for block in values["blocks"])
        return cls(**values)
