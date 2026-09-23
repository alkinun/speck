"""Define the block grammar for Speck model architectures."""

import json
import math
from dataclasses import asdict, dataclass, field, replace


def canonical_json(value):
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _integer_fields(config, *names):
    """Reject boolean and fractional dimensions before shape arithmetic."""

    for name in names:
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{type(config).__name__}.{name} must be an integer")


def _validate_delta_geometry(spec, name):
    dimensions = ("key_head_dim", "value_head_dim", "num_key_heads", "num_value_heads")
    _integer_fields(spec, *dimensions, "conv_kernel_size")
    if any(getattr(spec, field) < 1 for field in dimensions):
        raise ValueError(f"{name} dimensions and head counts must be positive")
    if spec.num_value_heads % spec.num_key_heads:
        raise ValueError(f"{name} value heads must be divisible by key heads")
    if spec.conv_kernel_size < 2:
        raise ValueError(f"{name} convolution kernels need at least two positions")


@dataclass(frozen=True)
class AttentionSpec:
    head_dim: int
    num_key_value_heads: int
    rope_dim: int | None = None
    kind: str = field(init=False, default="attention")

    @property
    def active_rope_dim(self):
        return self.head_dim if self.rope_dim is None else self.rope_dim

    def __post_init__(self):
        _integer_fields(self, "head_dim", "num_key_value_heads")
        if self.rope_dim is not None:
            _integer_fields(self, "rope_dim")
        if self.head_dim < 2 or self.head_dim % 2:
            raise ValueError("attention head dimensions must be positive and even")
        if self.num_key_value_heads < 1:
            raise ValueError("attention KV heads must be positive")
        rope_dim = self.active_rope_dim
        if rope_dim < 0 or rope_dim > self.head_dim or rope_dim % 2:
            raise ValueError("attention RoPE dimensions must be even and within the head")


# Checkpoint metadata written before these options were removed records them at their defaults.
_LEGACY_ATTENTION_DEFAULTS = {
    "scope": "global",
    "window_size": None,
    "output_gate": "none",
    "memory": None,
    "memory_role": "none",
}


@dataclass(frozen=True)
class KimiDeltaAttentionSpec:
    key_head_dim: int
    value_head_dim: int
    num_key_heads: int
    num_value_heads: int
    conv_kernel_size: int = 4
    output_gate_activation: str = "sigmoid"
    kind: str = field(init=False, default="kimi_delta_attention")

    def __post_init__(self):
        _validate_delta_geometry(self, "Kimi Delta Attention")
        if self.key_head_dim != self.value_head_dim:
            raise ValueError("Kimi Delta Attention requires equal key and value head dimensions")
        if self.output_gate_activation not in {"sigmoid", "silu"}:
            raise ValueError("Kimi Delta Attention output gate activation must be sigmoid or silu")


@dataclass(frozen=True)
class SwiGLUSpec:
    intermediate_size: int
    kind: str = field(init=False, default="swiglu")

    def __post_init__(self):
        _integer_fields(self, "intermediate_size")
        if self.intermediate_size < 1:
            raise ValueError("SwiGLU intermediate sizes must be positive")


OperationSpec = AttentionSpec | KimiDeltaAttentionSpec | SwiGLUSpec


def operation_from_dict(value):
    value = dict(value)
    kind = value.pop("kind")
    if kind == "attention":
        for key, default in _LEGACY_ATTENTION_DEFAULTS.items():
            if value.pop(key, default) != default:
                raise ValueError(f"unsupported attention option: {key}")
    classes = {
        "attention": AttentionSpec,
        "kimi_delta_attention": KimiDeltaAttentionSpec,
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
        _integer_fields(self, "hidden_size")
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
        _integer_fields(self, "repeat")
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
    expected_active_parameters: int | None = None
    tie_word_embeddings: bool = True

    def __post_init__(self):
        _integer_fields(
            self,
            "embedding_size",
            "vocab_size",
            "max_position_embeddings",
            "bos_token_id",
            "eos_token_id",
        )
        for name in ("expected_parameters", "expected_active_parameters"):
            if getattr(self, name) is not None:
                _integer_fields(self, name)
        if not self.blocks:
            raise ValueError("architectures need at least one block")
        if not isinstance(self.tie_word_embeddings, bool):
            raise ValueError("tie_word_embeddings must be boolean")
        if not self.tie_word_embeddings:
            raise ValueError("untied word embeddings are not supported")
        if self.embedding_size < 1 or self.vocab_size < 1:
            raise ValueError("embedding and vocabulary sizes must be positive")
        if self.max_position_embeddings < 1:
            raise ValueError("maximum positions must be positive")
        for name in ("rope_theta", "rope_scaling_factor", "rms_norm_eps", "initializer_range"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
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
        if self.expected_active_parameters is not None:
            if self.expected_active_parameters < 1:
                raise ValueError("expected active parameters must be positive")
            if (
                self.expected_parameters is not None
                and self.expected_active_parameters > self.expected_parameters
            ):
                raise ValueError("expected active parameters cannot exceed total parameters")

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
        for group in values["blocks"]:
            for stage in group["block"]["stages"]:
                for branch in stage["branches"]:
                    if (
                        branch["kind"] == "kimi_delta_attention"
                        and branch["output_gate_activation"] == "sigmoid"
                    ):
                        branch.pop("output_gate_activation")
        values.pop("expected_parameters")
        values.pop("expected_active_parameters")
        return json.loads(canonical_json(values))

    def export(self):
        values = self.settings()
        if self.expected_parameters is not None:
            values["expected_parameters"] = self.expected_parameters
        if self.expected_active_parameters is not None:
            values["expected_active_parameters"] = self.expected_active_parameters
        return values

    def active_parameter_count(self, total_parameters):
        """Return per-token parameters; every parameter is active in a dense model."""

        if (
            isinstance(total_parameters, bool)
            or not isinstance(total_parameters, int)
            or total_parameters < 1
        ):
            raise ValueError("total parameters must be a positive integer")
        return total_parameters

    @classmethod
    def from_dict(cls, value):
        values = dict(value)
        values["blocks"] = tuple(BlockGroup.from_dict(block) for block in values["blocks"])
        return cls(**values)
