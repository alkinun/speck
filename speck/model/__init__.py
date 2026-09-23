"""Implement the Speck hybrid decoder language model."""

from collections import defaultdict
from dataclasses import replace

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from speck.model.architecture import (
    ArchitectureConfig,
    AttentionSpec,
    KimiDeltaAttentionSpec,
    SwiGLUSpec,
)
from speck.model.layers import (
    _LOSS_BACKENDS,
    Attention,
    KimiDeltaAttention,
    Linear,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    attention_rotary_key,
    initialize_delta_timescales,
    linear_cross_entropy,
)
from speck.model.state import AttentionState, DeltaNetState, SequenceState
from speck.training.optimizers import BatchedMuon, CombinedOptimizer, DeviceAdamW


class Operation(nn.Module):
    def __init__(self, hidden_size, spec, config):
        super().__init__()
        self.spec = spec
        self.norm = RMSNorm(hidden_size, config.rms_norm_eps)
        if isinstance(spec, AttentionSpec):
            self.operation = Attention(hidden_size, spec, config.rms_norm_eps)
        elif isinstance(spec, KimiDeltaAttentionSpec):
            self.operation = KimiDeltaAttention(hidden_size, spec, config.rms_norm_eps)
        elif isinstance(spec, SwiGLUSpec):
            self.operation = SwiGLU(hidden_size, spec)
        else:
            raise TypeError("unsupported architecture operation")

    def forward(self, x, rotary, position, state=None):
        normalized = self.norm(x)
        if isinstance(self.spec, AttentionSpec):
            embedding = (
                rotary[attention_rotary_key(self.spec)] if self.spec.active_rope_dim else None
            )
            return self.operation(normalized, embedding, position, state)
        if isinstance(self.spec, KimiDeltaAttentionSpec):
            return self.operation(normalized, state)
        return self.operation(normalized)


class Stage(nn.Module):
    def __init__(self, hidden_size, config, stage_index, stage):
        super().__init__()
        self.stage_index = stage_index
        self.branches = nn.ModuleList(
            Operation(hidden_size, spec, config) for spec in stage.branches
        )

    def forward(self, x, rotary, position, state, occurrence):
        outputs = []
        for branch_index, branch in enumerate(self.branches):
            key = f"occurrence_{occurrence}_stage_{self.stage_index}_branch_{branch_index}"
            entry = state.entries[key] if state is not None and key in state.entries else None
            outputs.append(branch(x, rotary, position, entry))
        return x + sum(outputs)


class BlockCore(nn.Module):
    def __init__(self, block, config):
        super().__init__()
        self.stages = nn.ModuleList(
            Stage(block.hidden_size, config, index, stage)
            for index, stage in enumerate(block.stages)
        )

    def forward(self, x, rotary, position, state, occurrence):
        for stage in self.stages:
            x = stage(x, rotary, position, state, occurrence)
        return x


class SpeckForCausalLM(nn.Module):
    """Implement the configurable Speck causal language model."""

    def __init__(self, config, loss_backend="torch"):
        super().__init__()
        if not isinstance(config, ArchitectureConfig):
            raise TypeError("model requires an architecture config")
        if loss_backend not in _LOSS_BACKENDS:
            raise ValueError(f"unsupported loss backend: {loss_backend}")
        self.config = config
        self.loss_backend = loss_backend
        self.gradient_checkpointing = False
        self.embed_tokens = nn.Embedding(config.vocab_size, config.embedding_size)
        self.execution_plan = config.execution_plan
        self.cores = nn.ModuleDict()
        for invocation in self.execution_plan:
            if invocation.weight_key not in self.cores:
                self.cores[invocation.weight_key] = BlockCore(invocation.block, config)
        adapters = []
        input_size = config.embedding_size
        for invocation in self.execution_plan:
            output_size = invocation.block.hidden_size
            adapters.append(
                Linear(input_size, output_size, bias=False)
                if input_size != output_size
                else nn.Identity()
            )
            input_size = output_size
        self.adapters = nn.ModuleList(adapters)
        self.norm = RMSNorm(input_size, config.rms_norm_eps)
        self.output_projection = (
            Linear(input_size, config.embedding_size, bias=False)
            if input_size != config.embedding_size
            else nn.Identity()
        )
        self.lm_head = Linear(config.embedding_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        self.rotary = nn.ModuleDict(
            {
                attention_rotary_key(branch): RotaryEmbedding(
                    branch.active_rope_dim, config.rope_theta, config.rope_scaling_factor
                )
                for invocation in self.execution_plan
                for stage in invocation.block.stages
                for branch in stage.branches
                if isinstance(branch, AttentionSpec) and branch.active_rope_dim > 0
            }
        )

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Strictly preserve the physical embedding/head tie across checkpoint loads."""

        embedding = state_dict.get("embed_tokens.weight")
        head = state_dict.get("lm_head.weight")
        if embedding is not None and head is not None and not torch.equal(embedding, head):
            raise RuntimeError("checkpoint input embedding and LM head tensors are not tied")
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.lm_head.weight = self.embed_tokens.weight
        return result

    @torch.no_grad()
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, (Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=self.config.initializer_range)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
            elif isinstance(module, KimiDeltaAttention):
                nn.init.normal_(module.conv_kernel, std=self.config.initializer_range)
                initialize_delta_timescales(
                    module.log_rates,
                    module.decay_bias,
                    minimum_rate=1.0,
                )

    @torch.no_grad()
    def resize_token_embeddings(self, vocab_size):
        """Grow tied token embeddings while preserving all pretrained rows."""

        current = self.config.vocab_size
        if vocab_size < current:
            raise ValueError("token embedding resize cannot shrink the vocabulary")
        if vocab_size == current:
            return self.embed_tokens
        embedding = nn.Embedding(
            vocab_size,
            self.config.embedding_size,
            device=self.embed_tokens.weight.device,
            dtype=self.embed_tokens.weight.dtype,
        )
        nn.init.normal_(embedding.weight, std=self.config.initializer_range)
        embedding.weight[:current].copy_(self.embed_tokens.weight)
        self.embed_tokens = embedding
        self.lm_head = Linear(self.config.embedding_size, vocab_size, bias=False).to(
            device=embedding.weight.device,
            dtype=embedding.weight.dtype,
        )
        self.lm_head.weight = self.embed_tokens.weight
        self.config = replace(
            self.config,
            vocab_size=vocab_size,
            expected_parameters=None,
            expected_active_parameters=None,
        )
        return self.embed_tokens

    def set_gradient_checkpointing(self, enabled=True):
        if not isinstance(enabled, bool):
            raise TypeError("gradient checkpointing flag must be boolean")
        self.gradient_checkpointing = enabled

    def state(
        self,
        batch_size=1,
        length=None,
        device=None,
        dtype=None,
        kv_cache_dtype=None,
    ):
        parameter = next(self.parameters())
        device = torch.device(device or parameter.device)
        dtype = dtype or (torch.bfloat16 if device.type == "cuda" else parameter.dtype)
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("state batch size must be a positive integer")
        if length is None:
            length = self.config.max_position_embeddings
        if (
            not isinstance(length, int)
            or isinstance(length, bool)
            or length < 1
            or length > self.config.max_position_embeddings
        ):
            raise ValueError("state length is outside the model context")
        entries = {}
        for invocation in self.execution_plan:
            for stage_index, stage in enumerate(invocation.block.stages):
                for branch_index, branch in enumerate(stage.branches):
                    key = f"occurrence_{invocation.occurrence_index}_stage_{stage_index}_branch_{branch_index}"
                    if isinstance(branch, AttentionSpec):
                        entries[key] = AttentionState(
                            batch_size,
                            branch.num_key_value_heads,
                            length,
                            branch.head_dim,
                            device,
                            dtype,
                            storage_dtype=kv_cache_dtype,
                        )
                    elif isinstance(branch, KimiDeltaAttentionSpec):
                        key_dim = branch.num_key_heads * branch.key_head_dim
                        value_dim = branch.num_value_heads * branch.value_head_dim
                        entries[key] = DeltaNetState(
                            batch_size,
                            branch.num_value_heads,
                            branch.key_head_dim,
                            branch.value_head_dim,
                            2 * key_dim + value_dim,
                            branch.conv_kernel_size - 1,
                            device,
                            dtype,
                            kind=branch.kind,
                        )
        return SequenceState(entries, length)

    def forward(
        self,
        tokens=None,
        targets=None,
        state=None,
        inputs_embeds=None,
        return_hidden=False,
        last_token_only=False,
        loss_reduction="mean",
    ):
        if (tokens is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of tokens or inputs_embeds")
        if targets is not None and last_token_only:
            raise ValueError("last-token logits cannot be used with full-sequence targets")
        x = self.embed_tokens(tokens) if inputs_embeds is None else inputs_embeds
        x = x.to(torch.bfloat16 if x.is_cuda else self.embed_tokens.weight.dtype)
        length = x.size(1)
        position = state.position if state is not None else 0
        maximum = state.length if state is not None else self.config.max_position_embeddings
        if position + length > maximum:
            raise ValueError("sequence exceeds the available model state")
        for invocation, adapter in zip(self.execution_plan, self.adapters):
            x = adapter(x)
            core = self.cores[invocation.weight_key]
            if self.training and self.gradient_checkpointing and state is None:
                x = activation_checkpoint(
                    core,
                    x,
                    self.rotary,
                    position,
                    None,
                    invocation.occurrence_index,
                    use_reentrant=False,
                )
            else:
                x = core(x, self.rotary, position, state, invocation.occurrence_index)
        if state is not None:
            state.position += length
        hidden = self.output_projection(self.norm(x))
        if targets is not None:
            output = linear_cross_entropy(
                hidden,
                self.lm_head.weight,
                targets,
                loss_reduction,
                self.loss_backend,
            )
        else:
            output = self.lm_head(hidden[:, -1:] if last_token_only else hidden).float()
        return (output, hidden) if return_hidden else output

    def optimizer(self, lr=6e-4, weight_decay=0.1, name="adamw"):
        embedding = self.embed_tokens.weight
        matrices, other_decay, no_decay = [], [], []
        for parameter in self.parameters():
            if parameter is embedding or parameter.ndim < 2:
                no_decay.append(parameter)
            elif parameter.ndim == 2:
                matrices.append(parameter)
            else:
                other_decay.append(parameter)
        fused = embedding.device.type == "cuda"
        if name == "muon":
            return CombinedOptimizer(
                muon=BatchedMuon(
                    matrices,
                    lr=lr,
                    weight_decay=weight_decay,
                    adjust_lr_fn="match_rms_adamw",
                ),
                adamw=DeviceAdamW(
                    [
                        {"params": other_decay, "weight_decay": weight_decay},
                        {"params": no_decay, "weight_decay": 0.0},
                    ],
                    lr=lr,
                    betas=(0.9, 0.95),
                    eps=1e-8,
                    fused=fused,
                ),
            )
        if name != "adamw":
            raise ValueError(f"unsupported optimizer: {name}")
        return DeviceAdamW(
            [
                {"params": matrices + other_decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=fused,
        )

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def active_parameter_count(self):
        return self.config.active_parameter_count(self.parameter_count())

    def optimizer_role_counts(self, optimizer):
        """Audit exact optimizer membership and summarize tensor/element roles."""

        parameter_ids = {id(parameter) for parameter in self.parameters()}
        memberships = {}
        roles = defaultdict(list)
        optimizers = (
            optimizer.optimizers.items()
            if isinstance(optimizer, CombinedOptimizer)
            else (("adamw", optimizer),)
        )
        for optimizer_name, member in optimizers:
            for group in member.param_groups:
                if optimizer_name == "muon":
                    role = "muon"
                else:
                    role = "adamw_decay" if group.get("weight_decay", 0.0) else "adamw_no_decay"
                for parameter in group["params"]:
                    identifier = id(parameter)
                    if identifier in memberships:
                        raise ValueError("a parameter appears in more than one optimizer role")
                    memberships[identifier] = role
                    roles[role].append(parameter)
        if set(memberships) != parameter_ids:
            raise ValueError("optimizer roles do not cover every model parameter exactly once")
        return {
            role: {
                "tensors": len(parameters),
                "parameters": sum(parameter.numel() for parameter in parameters),
            }
            for role, parameters in sorted(roles.items())
        }

    def flops_per_token(self, sequence_length):
        linear = self.config.vocab_size * self.config.embedding_size
        input_size = self.config.embedding_size
        attention = 0
        for invocation in self.execution_plan:
            hidden_size = invocation.block.hidden_size
            if input_size != hidden_size:
                linear += input_size * hidden_size
            for stage in invocation.block.stages:
                for branch in stage.branches:
                    if isinstance(branch, AttentionSpec):
                        kv_size = branch.num_key_value_heads * branch.head_dim
                        linear += 2 * hidden_size * hidden_size + 2 * hidden_size * kv_size
                        # Mean attended keys per query under a causal mask.
                        attention += 12 * (sequence_length + 1) / 2 * hidden_size
                    elif isinstance(branch, KimiDeltaAttentionSpec):
                        key_size = branch.num_key_heads * branch.key_head_dim
                        value_size = branch.num_value_heads * branch.value_head_dim
                        gate_rank = branch.value_head_dim
                        linear += hidden_size * (2 * key_size + 3 * value_size)
                        linear += hidden_size * (branch.num_value_heads + gate_rank)
                        linear += gate_rank * branch.num_value_heads * branch.key_head_dim
                        linear += (2 * key_size + value_size) * branch.conv_kernel_size
                        chunk_size = min(64, sequence_length)
                        head_dim = branch.key_head_dim
                        attention += branch.num_value_heads * (
                            6 * head_dim**2 + 3 * chunk_size * head_dim + chunk_size**2
                        )
                    elif isinstance(branch, SwiGLUSpec):
                        linear += 3 * hidden_size * branch.intermediate_size
            input_size = hidden_size
        if input_size != self.config.embedding_size:
            linear += input_size * self.config.embedding_size
        return 6 * linear + attention


def build_model(settings, vocab_size, bos_token_id=1, eos_token_id=2, loss_backend="torch"):
    values = dict(settings)
    model_vocab_size = values.get("vocab_size", vocab_size)
    if type(model_vocab_size) is not int or model_vocab_size < vocab_size:
        raise ValueError("model vocabulary must cover the tokenizer vocabulary")
    values.update(
        vocab_size=model_vocab_size,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
    )
    config = ArchitectureConfig.from_dict(values)
    model = SpeckForCausalLM(config, loss_backend=loss_backend)
    if config.expected_parameters is not None:
        actual = model.parameter_count()
        if actual != config.expected_parameters:
            raise ValueError(
                f"expected {config.expected_parameters:,} parameters but built {actual:,}"
            )
    if config.expected_active_parameters is not None:
        actual = model.active_parameter_count()
        if actual != config.expected_active_parameters:
            raise ValueError(
                "expected "
                f"{config.expected_active_parameters:,} active parameters but built {actual:,}"
            )
    return model
