"""focused hybrid runtime for version three speck architectures."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from speck.architecture import (
    ArchitectureConfig,
    AttentionSpec,
    GatedCausalConvSpec,
    SwiGLUSpec,
)
from speck.model import CombinedOptimizer, Linear, RMSNorm, rotate


class AttentionState:
    def __init__(self, batch_size, kv_heads, capacity, head_dim, device, dtype):
        if capacity < 1:
            raise ValueError("attention state capacity must be positive")
        shape = (batch_size, kv_heads, capacity, head_dim)
        self.keys = torch.empty(shape, device=device, dtype=dtype)
        self.values = torch.empty(shape, device=device, dtype=dtype)
        self.capacity = capacity
        self.used = 0
        self.write_position = 0

    def current(self):
        if self.used == 0:
            return self.keys[:, :, :0], self.values[:, :, :0]
        if self.used < self.capacity:
            return self.keys[:, :, :self.used], self.values[:, :, :self.used]
        if self.write_position == 0:
            return self.keys, self.values
        keys = torch.cat(
            (self.keys[:, :, self.write_position:], self.keys[:, :, :self.write_position]),
            dim=2,
        )
        values = torch.cat(
            (
                self.values[:, :, self.write_position:],
                self.values[:, :, :self.write_position],
            ),
            dim=2,
        )
        return keys, values

    def append(self, keys, values):
        length = keys.size(2)
        if length >= self.capacity:
            self.keys.copy_(keys[:, :, -self.capacity:])
            self.values.copy_(values[:, :, -self.capacity:])
            self.used = self.capacity
            self.write_position = 0
            return
        first = min(length, self.capacity - self.write_position)
        end = self.write_position + first
        self.keys[:, :, self.write_position:end] = keys[:, :, :first]
        self.values[:, :, self.write_position:end] = values[:, :, :first]
        remaining = length - first
        if remaining:
            self.keys[:, :, :remaining] = keys[:, :, first:]
            self.values[:, :, :remaining] = values[:, :, first:]
        self.write_position = (self.write_position + length) % self.capacity
        self.used = min(self.capacity, self.used + length)

    def allocated_bytes(self):
        return sum(tensor.numel() * tensor.element_size() for tensor in (self.keys, self.values))


class ConvolutionState:
    def __init__(self, batch_size, inner_size, history, device, dtype):
        self.values = torch.zeros(
            batch_size,
            inner_size,
            history,
            device=device,
            dtype=dtype,
        )

    def allocated_bytes(self):
        return self.values.numel() * self.values.element_size()


class SequenceState:
    def __init__(self, entries, length):
        self.entries = entries
        self.position = 0
        self.length = length

    def reset(self):
        self.position = 0
        for entry in self.entries.values():
            if isinstance(entry, AttentionState):
                entry.used = 0
                entry.write_position = 0
            else:
                entry.values.zero_()

    def allocated_bytes(self):
        return sum(entry.allocated_bytes() for entry in self.entries.values())


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, length, theta):
        super().__init__()
        frequency = 1 / (
            theta
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(length, dtype=torch.float32)
        angles = torch.outer(positions, frequency).repeat(1, 2)
        self.register_buffer("cos", angles.cos()[None, None], persistent=False)
        self.register_buffer("sin", angles.sin()[None, None], persistent=False)

    def forward(self, position, length, dtype):
        end = position + length
        return (
            self.cos[..., position:end, :].to(dtype),
            self.sin[..., position:end, :].to(dtype),
        )


class Attention(nn.Module):
    def __init__(self, hidden_size, spec, eps):
        super().__init__()
        self.spec = spec
        self.q_heads = hidden_size // spec.head_dim
        kv_size = spec.num_key_value_heads * spec.head_dim
        self.q_proj = Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = Linear(hidden_size, kv_size, bias=False)
        self.v_proj = Linear(hidden_size, kv_size, bias=False)
        self.o_proj = Linear(hidden_size, hidden_size, bias=False)
        self.q_norm = RMSNorm(spec.head_dim, eps)
        self.k_norm = RMSNorm(spec.head_dim, eps)

    def forward(self, x, rotary, position, state=None):
        batch, length, hidden_size = x.shape
        q = self.q_proj(x).view(
            batch, length, self.q_heads, self.spec.head_dim
        ).transpose(1, 2)
        k = self.k_proj(x).view(
            batch,
            length,
            self.spec.num_key_value_heads,
            self.spec.head_dim,
        ).transpose(1, 2)
        v = self.v_proj(x).view(
            batch,
            length,
            self.spec.num_key_value_heads,
            self.spec.head_dim,
        ).transpose(1, 2)
        cos, sin = rotary(position, length, q.dtype)
        q = rotate(self.q_norm(q), cos, sin)
        k = rotate(self.k_norm(k), cos, sin)

        if state is None and self.spec.scope == "global":
            keys, values = k, v
            mask = None
            causal = True
        else:
            if state is None:
                past_k = k[:, :, :0]
                past_v = v[:, :, :0]
            else:
                past_k, past_v = state.current()
            keys = torch.cat((past_k, k), dim=2)
            values = torch.cat((past_v, v), dim=2)
            past_length = past_k.size(2)
            key_start = position - past_length
            key_positions = torch.arange(
                key_start,
                position + length,
                device=x.device,
            )
            query_positions = torch.arange(
                position,
                position + length,
                device=x.device,
            )[:, None]
            mask = key_positions[None, :] <= query_positions
            if self.spec.scope == "sliding":
                mask &= key_positions[None, :] > query_positions - self.spec.window_size
            causal = False
        output = F.scaled_dot_product_attention(
            q,
            keys,
            values,
            attn_mask=mask,
            is_causal=causal,
            enable_gqa=self.q_heads != self.spec.num_key_value_heads,
        )
        if state is not None:
            state.append(k, v)
        return self.o_proj(output.transpose(1, 2).contiguous().view(batch, length, hidden_size))


class GatedCausalConv(nn.Module):
    def __init__(self, hidden_size, spec):
        super().__init__()
        self.spec = spec
        self.input_projection = Linear(hidden_size, 3 * spec.inner_size, bias=False)
        self.kernel = nn.Parameter(torch.empty(spec.inner_size, 1, spec.kernel_size))
        self.output_projection = Linear(spec.inner_size, hidden_size, bias=False)

    def forward(self, x, state=None):
        first_gate, second_gate, values = self.input_projection(x).chunk(3, dim=-1)
        values = first_gate * values
        transposed = values.transpose(1, 2)
        if state is None:
            transposed = F.pad(transposed, (self.spec.kernel_size - 1, 0))
        else:
            transposed = torch.cat((state.values, transposed), dim=2)
        convolved = F.conv1d(transposed, self.kernel, groups=self.spec.inner_size)
        if state is not None:
            state.values.copy_(transposed[:, :, -(self.spec.kernel_size - 1):])
        return self.output_projection((second_gate * convolved.transpose(1, 2)))


class SwiGLU(nn.Module):
    def __init__(self, hidden_size, spec):
        super().__init__()
        self.gate_proj = Linear(hidden_size, spec.intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, spec.intermediate_size, bias=False)
        self.down_proj = Linear(spec.intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Operation(nn.Module):
    def __init__(self, hidden_size, spec, config):
        super().__init__()
        self.spec = spec
        self.norm = RMSNorm(hidden_size, config.rms_norm_eps)
        if isinstance(spec, AttentionSpec):
            self.operation = Attention(hidden_size, spec, config.rms_norm_eps)
        elif isinstance(spec, GatedCausalConvSpec):
            self.operation = GatedCausalConv(hidden_size, spec)
        elif isinstance(spec, SwiGLUSpec):
            self.operation = SwiGLU(hidden_size, spec)
        else:
            raise TypeError("unsupported architecture operation")

    def forward(self, x, rotary, position, state=None):
        normalized = self.norm(x)
        if isinstance(self.spec, AttentionSpec):
            return self.operation(
                normalized,
                rotary[str(self.spec.head_dim)],
                position,
                state,
            )
        if isinstance(self.spec, GatedCausalConvSpec):
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
        residual = x
        outputs = []
        for branch_index, branch in enumerate(self.branches):
            key = f"occurrence_{occurrence}_stage_{self.stage_index}_branch_{branch_index}"
            entry = state.entries[key] if state is not None and key in state.entries else None
            outputs.append(branch(residual, rotary, position, entry))
        return residual + sum(outputs)


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


class SpeckV3ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        if not isinstance(config, ArchitectureConfig):
            raise TypeError("v3 model requires an architecture config")
        self.config = config
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
        head_dimensions = {
            branch.head_dim
            for invocation in self.execution_plan
            for stage in invocation.block.stages
            for branch in stage.branches
            if isinstance(branch, AttentionSpec)
        }
        self.rotary = nn.ModuleDict(
            {
                str(head_dim): RotaryEmbedding(
                    head_dim,
                    config.max_position_embeddings,
                    config.rope_theta,
                )
                for head_dim in head_dimensions
            }
        )

    @torch.no_grad()
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, (Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=self.config.initializer_range)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
            elif isinstance(module, GatedCausalConv):
                nn.init.normal_(module.kernel, std=self.config.initializer_range)

    def state(self, batch_size=1, length=None, device=None, dtype=None):
        parameter = next(self.parameters())
        device = torch.device(device or parameter.device)
        dtype = dtype or parameter.dtype
        length = length or self.config.max_position_embeddings
        if length < 1 or length > self.config.max_position_embeddings:
            raise ValueError("state length is outside the model context")
        entries = {}
        for invocation in self.execution_plan:
            for stage_index, stage in enumerate(invocation.block.stages):
                for branch_index, branch in enumerate(stage.branches):
                    key = (
                        f"occurrence_{invocation.occurrence_index}_stage_{stage_index}"
                        f"_branch_{branch_index}"
                    )
                    if isinstance(branch, AttentionSpec):
                        capacity = length if branch.scope == "global" else min(
                            length, branch.window_size
                        )
                        entries[key] = AttentionState(
                            batch_size,
                            branch.num_key_value_heads,
                            capacity,
                            branch.head_dim,
                            device,
                            dtype,
                        )
                    elif isinstance(branch, GatedCausalConvSpec):
                        entries[key] = ConvolutionState(
                            batch_size,
                            branch.inner_size,
                            branch.kernel_size - 1,
                            device,
                            dtype,
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
    ):
        if (tokens is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of tokens or inputs embeds")
        if targets is not None and last_token_only:
            raise ValueError("last token logits cannot be used with full sequence targets")
        if inputs_embeds is None:
            x = self.embed_tokens(tokens)
        else:
            x = inputs_embeds
        x = x.to(self.embed_tokens.weight.dtype)
        length = x.size(1)
        position = state.position if state is not None else 0
        maximum = state.length if state is not None else self.config.max_position_embeddings
        if position + length > maximum:
            raise ValueError("sequence exceeds the available model state")
        for invocation, adapter in zip(self.execution_plan, self.adapters):
            x = adapter(x)
            x = self.cores[invocation.weight_key](
                x,
                self.rotary,
                position,
                state,
                invocation.occurrence_index,
            )
        if state is not None:
            state.position += length
        hidden = self.output_projection(self.norm(x))
        logits_input = hidden[:, -1:] if last_token_only else hidden
        logits = self.lm_head(logits_input).float()
        output = (
            F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            if targets is not None
            else logits
        )
        return (output, hidden) if return_hidden else output

    def optimizer(self, lr=6e-4, weight_decay=0.1, name="adamw"):
        embedding = self.embed_tokens.weight
        decay, no_decay = [], []
        for parameter in self.parameters():
            (no_decay if parameter is embedding or parameter.ndim < 2 else decay).append(parameter)
        if name == "muon":
            return CombinedOptimizer(
                muon=torch.optim.Muon(
                    decay,
                    lr=lr,
                    weight_decay=weight_decay,
                    adjust_lr_fn="match_rms_adamw",
                ),
                adamw=torch.optim.AdamW(
                    no_decay,
                    lr=lr,
                    weight_decay=0.0,
                    betas=(0.9, 0.95),
                    eps=1e-8,
                ),
            )
        if name != "adamw":
            raise ValueError(f"unsupported optimizer: {name}")
        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

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
                        linear += 2 * hidden_size * hidden_size
                        linear += 2 * hidden_size * kv_size
                        context = sequence_length if branch.scope == "global" else min(
                            sequence_length, branch.window_size
                        )
                        attention += 12 * context * hidden_size
                    elif isinstance(branch, GatedCausalConvSpec):
                        linear += 4 * hidden_size * branch.inner_size
                        linear += branch.inner_size * branch.kernel_size
                    else:
                        linear += 3 * hidden_size * branch.intermediate_size
            input_size = hidden_size
        if input_size != self.config.embedding_size:
            linear += input_size * self.config.embedding_size
        return 6 * linear + attention
