"""Implement the Speck hybrid decoder language model."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right
from torch.nn.attention.flex_attention import BlockMask, flex_attention

_LOSS_BACKENDS = {"torch", "liger"}


_COMPILED_FLEX_ATTENTION = torch.compile(flex_attention, dynamic=False)


@torch.compiler.disable
def liger_linear_cross_entropy(hidden, weight, targets, reduction):
    try:
        from liger_kernel.transformers.functional import liger_fused_linear_cross_entropy
    except ImportError as exception:
        raise RuntimeError(
            "the Liger loss backend requires the GPU dependencies; run `uv sync --extra gpu`"
        ) from exception
    return liger_fused_linear_cross_entropy(
        hidden,
        weight,
        targets,
        reduction=reduction,
    )


def linear_cross_entropy(hidden, weight, targets, reduction, backend):
    hidden = hidden.flatten(0, 1)
    targets = targets.flatten()
    compute_weight = weight.to(hidden.dtype)
    if backend == "torch":
        logits = F.linear(hidden, compute_weight).float()
        return F.cross_entropy(logits, targets, reduction=reduction)
    return liger_linear_cross_entropy(hidden, compute_weight, targets, reduction)


class Linear(nn.Linear):
    def forward(self, input):
        bias = self.bias.to(input.dtype) if self.bias is not None else None
        return F.linear(input, self.weight.to(input.dtype), bias)


class RMSNorm(nn.Module):
    def __init__(self, size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):
        normalized = F.rms_norm(x.float(), (x.size(-1),), eps=self.eps).to(x.dtype)
        return normalized * self.weight.to(x.dtype)


def rotate(x, cos, sin, rotary_dim):
    if rotary_dim == 0:
        return x
    rotary, passthrough = x[..., :rotary_dim], x[..., rotary_dim:]
    first, second = rotary.chunk(2, dim=-1)
    rotary = rotary * cos + torch.cat((-second, first), dim=-1) * sin
    return torch.cat((rotary, passthrough), dim=-1)


def causal_depthwise_conv1d(x, weight):
    """Apply a tiny causal depthwise stencil without a generic convolution launch."""

    kernel = weight[:, 0].to(x.dtype)
    output = x * kernel[:, -1, None]
    for delay in range(1, kernel.size(1)):
        shifted = F.pad(x, (delay, 0))[:, :, :-delay]
        output = output + shifted * kernel[:, -1 - delay, None]
    return output


class RotaryEmbedding(nn.Module):
    """Generate RoPE chunks on demand instead of retaining position-sized tables."""

    def __init__(self, rotary_dim, theta, scaling_factor):
        super().__init__()
        self.rotary_dim = rotary_dim
        self.theta = theta
        self.scaling_factor = scaling_factor
        self.register_buffer(
            "frequency",
            torch.empty(rotary_dim // 2, dtype=torch.float32),
            persistent=False,
        )
        self.reset_frequency()

    @torch.no_grad()
    def reset_frequency(self):
        """Rebuild the derived buffer after meta-device checkpoint loading."""

        frequency = 1 / (
            self.theta
            ** (
                torch.arange(
                    0,
                    self.rotary_dim,
                    2,
                    device=self.frequency.device,
                    dtype=torch.float32,
                )
                / self.rotary_dim
            )
        )
        self.frequency.copy_(frequency.to(self.frequency.dtype))

    def forward(self, position, length, dtype):
        positions = torch.arange(
            position,
            position + length,
            device=self.frequency.device,
            dtype=torch.float32,
        )
        positions = positions / self.scaling_factor
        angles = torch.outer(positions, self.frequency).repeat(1, 2)
        return angles.cos()[None, None].to(dtype), angles.sin()[None, None].to(dtype)


def attention_rotary_key(spec):
    rotary_dim = spec.head_dim if spec.rope_dim is None else spec.rope_dim
    return f"{spec.scope}:{spec.head_dim}:{rotary_dim}"


def causal_attention_mask(query_positions, key_positions, window_size=None):
    """Return the causal attention relation, optionally bounded to a local window."""

    mask = key_positions <= query_positions
    if window_size is not None:
        mask = mask & (key_positions > query_positions - window_size)
    return mask


def mean_causal_attention_context(sequence_length, window_size=None):
    """Return the mean number of attended keys per query token."""

    context = sequence_length if window_size is None else min(sequence_length, window_size)
    attended_pairs = context * (context + 1) // 2
    attended_pairs += (sequence_length - context) * context
    return attended_pairs / sequence_length


def torch_sliding_window_attention(query, key, value, position, past_length, window_size):
    """Reference local attention using explicit masks in bounded query chunks."""

    outputs = []
    query_chunk_size = min(2_048, max(256, window_size))
    first_key_position = position - past_length
    for start in range(0, query.size(2), query_chunk_size):
        end = min(start + query_chunk_size, query.size(2))
        query_start_position = position + start
        query_end_position = position + end
        key_start_position = max(first_key_position, query_start_position - window_size + 1)
        key_end_position = query_end_position
        key_start = key_start_position - first_key_position
        key_end = key_end_position - first_key_position
        selected_key = key[:, :, key_start:key_end]
        selected_value = value[:, :, key_start:key_end]
        query_positions = torch.arange(
            query_start_position, query_end_position, device=query.device
        )[:, None]
        key_positions = torch.arange(key_start_position, key_end_position, device=query.device)[
            None, :
        ]
        mask = causal_attention_mask(query_positions, key_positions, window_size)
        outputs.append(
            F.scaled_dot_product_attention(
                query[:, :, start:end],
                selected_key,
                selected_value,
                attn_mask=mask,
                enable_gqa=query.size(1) != key.size(1),
            )
        )
    return torch.cat(outputs, dim=2)


_SLIDING_WINDOW_BLOCK_MASKS = {}


_FLEX_BLOCK_SIZE = 128


def ordered_block_rows(rows, width, device):
    """Encode ordered block-index rows without materializing a token-level mask."""

    counts = torch.tensor([len(row) for row in rows], dtype=torch.int32, device=device)
    padded = [row + [0] * (width - len(row)) for row in rows]
    indices = torch.tensor(padded, dtype=torch.int32, device=device)
    return counts[None, None], indices[None, None]


@torch.compiler.assume_constant_result
def sliding_window_block_mask(device, query_length, key_length, past_length, window_size):
    """Build and cache a head-independent FlexAttention block mask."""

    cache_key = (device, query_length, key_length, past_length, window_size)
    if cache_key in _SLIDING_WINDOW_BLOCK_MASKS:
        return _SLIDING_WINDOW_BLOCK_MASKS[cache_key]

    def mask_mod(_batch, _head, query_index, key_index):
        return causal_attention_mask(query_index + past_length, key_index, window_size)

    query_blocks = (query_length + _FLEX_BLOCK_SIZE - 1) // _FLEX_BLOCK_SIZE
    key_blocks = (key_length + _FLEX_BLOCK_SIZE - 1) // _FLEX_BLOCK_SIZE
    partial_kv_rows = [[] for _ in range(query_blocks)]
    full_kv_rows = [[] for _ in range(query_blocks)]
    partial_q_rows = [[] for _ in range(key_blocks)]
    full_q_rows = [[] for _ in range(key_blocks)]
    for query_block in range(query_blocks):
        query_start = query_block * _FLEX_BLOCK_SIZE
        query_end = min(query_start + _FLEX_BLOCK_SIZE, query_length)
        first_key_block = max(
            0,
            (query_start + past_length - window_size + 1) // _FLEX_BLOCK_SIZE,
        )
        last_key_block = min(
            key_blocks - 1,
            (query_end - 1 + past_length) // _FLEX_BLOCK_SIZE,
        )
        for key_block in range(first_key_block, last_key_block + 1):
            key_start = key_block * _FLEX_BLOCK_SIZE
            key_end = min(key_start + _FLEX_BLOCK_SIZE, key_length)
            is_full = (
                query_end - query_start == _FLEX_BLOCK_SIZE
                and key_end - key_start == _FLEX_BLOCK_SIZE
                and key_end - 1 <= query_start + past_length
                and key_start > query_end - 1 + past_length - window_size
            )
            kv_rows = full_kv_rows if is_full else partial_kv_rows
            q_rows = full_q_rows if is_full else partial_q_rows
            kv_rows[query_block].append(key_block)
            q_rows[key_block].append(query_block)

    partial_kv_num, partial_kv_indices = ordered_block_rows(partial_kv_rows, key_blocks, device)
    full_kv_num, full_kv_indices = ordered_block_rows(full_kv_rows, key_blocks, device)
    partial_q_num, partial_q_indices = ordered_block_rows(partial_q_rows, query_blocks, device)
    full_q_num, full_q_indices = ordered_block_rows(full_q_rows, query_blocks, device)
    block_mask = BlockMask(
        seq_lengths=(query_length, key_length),
        kv_num_blocks=partial_kv_num,
        kv_indices=partial_kv_indices,
        full_kv_num_blocks=full_kv_num,
        full_kv_indices=full_kv_indices,
        q_num_blocks=partial_q_num,
        q_indices=partial_q_indices,
        full_q_num_blocks=full_q_num,
        full_q_indices=full_q_indices,
        BLOCK_SIZE=(_FLEX_BLOCK_SIZE, _FLEX_BLOCK_SIZE),
        mask_mod=mask_mod,
    )
    _SLIDING_WINDOW_BLOCK_MASKS[cache_key] = block_mask
    return block_mask


def flex_sliding_window_attention(query, key, value, past_length, window_size):
    """Evaluate local attention with a block-sparse FlexAttention kernel."""

    block_mask = sliding_window_block_mask(
        query.device,
        query.size(2),
        key.size(2),
        past_length,
        window_size,
    )
    return _COMPILED_FLEX_ATTENTION(
        query,
        key,
        value,
        block_mask=block_mask,
        enable_gqa=query.size(1) != key.size(1),
    )


class Attention(nn.Module):
    def __init__(self, hidden_size, spec, eps):
        super().__init__()
        self.spec = spec
        self.q_heads = hidden_size // spec.head_dim
        kv_size = spec.num_key_value_heads * spec.head_dim
        self.q_proj = Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = None if spec.reads_memory else Linear(hidden_size, kv_size, bias=False)
        self.v_proj = None if spec.reads_memory else Linear(hidden_size, kv_size, bias=False)
        self.o_proj = Linear(hidden_size, hidden_size, bias=False)
        if spec.output_gate == "headwise":
            self.gate_proj = Linear(hidden_size, self.q_heads, bias=False)
        elif spec.output_gate == "elementwise":
            self.gate_proj = Linear(hidden_size, hidden_size, bias=False)
        else:
            self.gate_proj = None
        self.q_norm = RMSNorm(spec.head_dim, eps)
        self.k_norm = None if spec.reads_memory else RMSNorm(spec.head_dim, eps)

    def forward(self, x, rotary, position, state=None, memory=None, produced=None):
        batch, length, hidden_size = x.shape
        q = self.q_proj(x).view(batch, length, self.q_heads, self.spec.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        rotary_dim = self.spec.active_rope_dim
        if rotary_dim:
            cos, sin = rotary(position, length, q.dtype)
            q = rotate(q, cos, sin, rotary_dim)
        if self.spec.reads_memory:
            if memory is None or self.spec.memory not in memory:
                raise ValueError(f"attention memory '{self.spec.memory}' has not been written")
            k = v = None
            keys, values = memory[self.spec.memory]
            past_length = keys.size(2) - length
            if past_length < 0:
                raise ValueError("attention memory is shorter than the reading query block")
        else:
            k = (
                self.k_proj(x)
                .view(batch, length, self.spec.num_key_value_heads, self.spec.head_dim)
                .transpose(1, 2)
            )
            v = (
                self.v_proj(x)
                .view(batch, length, self.spec.num_key_value_heads, self.spec.head_dim)
                .transpose(1, 2)
            )
            k = self.k_norm(k)
            if rotary_dim:
                k = rotate(k, cos, sin, rotary_dim)
            if state is None:
                keys, values, past_length = k, v, 0
            else:
                past_k, past_v = state.current()
                past_length = past_k.size(2)
                if past_length:
                    keys = torch.cat((past_k, k), dim=2)
                    values = torch.cat((past_v, v), dim=2)
                else:
                    keys, values = k, v
        if self.spec.scope != "global":
            mask, causal = None, False
        elif past_length == 0:
            mask, causal = None, True
        elif length == 1:
            mask, causal = None, False
        else:
            mask, causal = causal_lower_right(length, keys.size(2)), False
        if self.spec.writes_memory and produced is not None:
            produced[self.spec.memory] = (keys, values)
        if self.spec.scope == "sliding":
            assert self.spec.window_size is not None
            if q.is_cuda and q.size(-1) >= 16 and past_length == 0:
                output = flex_sliding_window_attention(
                    q,
                    keys,
                    values,
                    past_length,
                    self.spec.window_size,
                )
            else:
                output = torch_sliding_window_attention(
                    q,
                    keys,
                    values,
                    position,
                    past_length,
                    self.spec.window_size,
                )
        else:
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
        if self.gate_proj is not None:
            gate = self.gate_proj(x)
            gate_dim = 1 if self.spec.output_gate == "headwise" else self.spec.head_dim
            gate = gate.view(batch, length, self.q_heads, gate_dim).transpose(1, 2)
            output = output * gate.float().sigmoid().to(output.dtype)
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
        transposed = (first_gate * values).transpose(1, 2)
        history = self.spec.kernel_size - 1
        if state is not None:
            transposed = torch.cat((state.values, transposed), dim=2)
        convolved = causal_depthwise_conv1d(transposed, self.kernel)
        if state is not None:
            state.values.copy_(transposed[:, :, -history:])
            convolved = convolved[:, :, history:]
        return self.output_projection(second_gate * convolved.transpose(1, 2))


def torch_gated_delta_rule(query, key, value, log_decay, beta, initial_state=None):
    """Evaluate the gated delta rule exactly, retaining a differentiable reference path."""

    output_dtype = query.dtype
    query, key, value, log_decay, beta = (
        tensor.float() for tensor in (query, key, value, log_decay, beta)
    )
    query = query * torch.rsqrt(query.square().sum(dim=-1, keepdim=True) + 1e-6)
    key = key * torch.rsqrt(key.square().sum(dim=-1, keepdim=True) + 1e-6)
    query = query * (query.size(-1) ** -0.5)
    if initial_state is None:
        state = value.new_zeros(value.size(0), value.size(2), key.size(-1), value.size(-1))
    else:
        state = initial_state.float()
    outputs = []
    for index in range(query.size(1)):
        query_token = query[:, index]
        key_token = key[:, index]
        value_token = value[:, index]
        state = state * log_decay[:, index].exp()[..., None, None]
        remembered = torch.einsum("bhkv,bhk->bhv", state, key_token)
        delta = (value_token - remembered) * beta[:, index, :, None]
        state = state + torch.einsum("bhk,bhv->bhkv", key_token, delta)
        outputs.append(torch.einsum("bhkv,bhk->bhv", state, query_token))
    return torch.stack(outputs, dim=1).to(output_dtype), state


def gated_delta_rule(query, key, value, log_decay, beta, initial_state=None):
    """Use FLA on CUDA when available and the auditable Torch recurrence otherwise."""

    if query.is_cuda:
        try:
            from fla.ops.gated_delta_rule import (
                chunk_gated_delta_rule,
                fused_recurrent_gated_delta_rule,
            )
        except ImportError as error:
            raise RuntimeError(
                "CUDA Gated DeltaNet requires the pinned linear extra; "
                "use torch_gated_delta_rule directly for reference qualification"
            ) from error
        else:
            operation = (
                fused_recurrent_gated_delta_rule if query.size(1) == 1 else chunk_gated_delta_rule
            )
            return operation(
                query,
                key,
                value,
                g=log_decay,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
    return torch_gated_delta_rule(query, key, value, log_decay, beta, initial_state)


def torch_kimi_delta_rule(query, key, value, log_decay, beta, initial_state=None):
    """Evaluate channel-wise KDA exactly with a differentiable Torch recurrence."""

    output_dtype = query.dtype
    query, key, value, log_decay, beta = (
        tensor.float() for tensor in (query, key, value, log_decay, beta)
    )
    query = query * torch.rsqrt(query.square().sum(dim=-1, keepdim=True) + 1e-6)
    key = key * torch.rsqrt(key.square().sum(dim=-1, keepdim=True) + 1e-6)
    query = query * (query.size(-1) ** -0.5)
    if value.size(2) % query.size(2):
        raise ValueError("KDA value heads must be divisible by query and key heads")
    repetitions = value.size(2) // query.size(2)
    if repetitions > 1:
        query = query.repeat_interleave(repetitions, dim=2)
        key = key.repeat_interleave(repetitions, dim=2)
    if log_decay.shape != (*value.shape[:3], key.size(-1)):
        raise ValueError("KDA log decay must provide every value-head key channel")
    if beta.shape != value.shape[:3]:
        raise ValueError("KDA beta must provide every value head")
    if initial_state is None:
        state = value.new_zeros(value.size(0), value.size(2), key.size(-1), value.size(-1))
    else:
        state = initial_state.float()
    outputs = []
    for index in range(query.size(1)):
        query_token = query[:, index]
        key_token = key[:, index]
        value_token = value[:, index]
        state = state * log_decay[:, index].exp()[..., None]
        remembered = torch.einsum("bhkv,bhk->bhv", state, key_token)
        delta = (value_token - remembered) * beta[:, index, :, None]
        state = state + torch.einsum("bhk,bhv->bhkv", key_token, delta)
        outputs.append(torch.einsum("bhkv,bhk->bhv", state, query_token))
    return torch.stack(outputs, dim=1).to(output_dtype), state


def kda_log_decay(decay_logits, log_rates, decay_bias):
    """Return the KDA channel decay that the fused kernel computes internally."""

    heads, head_dim = decay_logits.shape[-2:]
    bias = decay_bias.view(heads, head_dim)
    return -log_rates.float().exp()[None, None, :, None] * F.softplus(decay_logits.float() + bias)


def kimi_delta_rule(
    query, key, value, decay_logits, beta, log_rates, decay_bias, initial_state=None
):
    """Use FLA KDA on CUDA and the auditable Torch recurrence elsewhere.

    The decay activation is fused into the CUDA kernel, which avoids materializing
    a full float32 gate tensor per layer. The Torch reference computes the same
    expression explicitly.
    """

    if query.is_cuda:
        try:
            from fla.ops.kda import chunk_kda, fused_recurrent_kda
        except ImportError as error:
            raise RuntimeError(
                "CUDA Kimi Delta Attention requires the pinned linear extra; "
                "use torch_kimi_delta_rule directly for reference qualification"
            ) from error
        operation = fused_recurrent_kda if query.size(1) == 1 else chunk_kda
        return operation(
            query,
            key,
            value,
            g=decay_logits,
            beta=beta,
            A_log=log_rates,
            dt_bias=decay_bias,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
        )
    log_decay = kda_log_decay(decay_logits, log_rates, decay_bias)
    return torch_kimi_delta_rule(query, key, value, log_decay, beta, initial_state)


@torch.no_grad()
def initialize_delta_timescales(log_rates, decay_bias, minimum_rate):
    """Initialize FLA-style recurrent rates and inverse-softplus time steps."""

    rates = torch.empty_like(log_rates, dtype=torch.float32).uniform_(minimum_rate, 16.0)
    rates.clamp_min_(1e-4)
    log_rates.copy_(rates.log().to(log_rates.dtype))
    log_minimum_dt = math.log(0.001)
    log_maximum_dt = math.log(0.1)
    dt = (
        torch.empty_like(decay_bias, dtype=torch.float32)
        .uniform_(
            log_minimum_dt,
            log_maximum_dt,
        )
        .exp_()
    )
    inverse_softplus = dt + torch.log(-torch.expm1(-dt))
    decay_bias.copy_(inverse_softplus.to(decay_bias.dtype))


class GatedDeltaNet(nn.Module):
    """A fixed-state, error-correcting linear sequence mixer with a local convolution."""

    def __init__(self, hidden_size, spec, eps):
        super().__init__()
        self.spec = spec
        self.key_dim = spec.num_key_heads * spec.key_head_dim
        self.value_dim = spec.num_value_heads * spec.value_head_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.qkvz_projection = Linear(
            hidden_size,
            2 * self.key_dim + 2 * self.value_dim,
            bias=False,
        )
        self.gates_projection = Linear(hidden_size, 2 * spec.num_value_heads, bias=False)
        self.conv_kernel = nn.Parameter(torch.empty(self.conv_dim, 1, spec.conv_kernel_size))
        if spec.decay_initialization == "speck":
            rates = torch.linspace(0.1, 1.0, spec.num_value_heads)
            decay_bias = torch.zeros(spec.num_value_heads)
        else:
            rates = torch.ones(spec.num_value_heads)
            decay_bias = torch.zeros(spec.num_value_heads)
        self.log_rates = nn.Parameter(rates.log())
        self.decay_bias = nn.Parameter(decay_bias)
        self.output_norm = RMSNorm(spec.value_head_dim, eps)
        self.output_projection = Linear(self.value_dim, hidden_size, bias=False)

    def forward(self, x, state=None):
        batch, length, _ = x.shape
        mixed = self.qkvz_projection(x)
        query, key, value, output_gate = torch.split(
            mixed,
            (self.key_dim, self.key_dim, self.value_dim, self.value_dim),
            dim=-1,
        )
        conv_input = torch.cat((query, key, value), dim=-1).transpose(1, 2)
        history = self.spec.conv_kernel_size - 1
        if state is not None:
            conv_input = torch.cat((state.convolution, conv_input), dim=2)
        convolved = F.silu(causal_depthwise_conv1d(conv_input, self.conv_kernel))
        if state is not None:
            state.convolution.copy_(conv_input[:, :, -history:].detach())
            convolved = convolved[:, :, history:]
        query, key, value = torch.split(
            convolved.transpose(1, 2),
            (self.key_dim, self.key_dim, self.value_dim),
            dim=-1,
        )
        query = query.view(batch, length, self.spec.num_key_heads, self.spec.key_head_dim)
        key = key.view(batch, length, self.spec.num_key_heads, self.spec.key_head_dim)
        value = value.view(batch, length, self.spec.num_value_heads, self.spec.value_head_dim)
        repetitions = self.spec.num_value_heads // self.spec.num_key_heads
        if repetitions > 1:
            query = query.repeat_interleave(repetitions, dim=2)
            key = key.repeat_interleave(repetitions, dim=2)
        beta_logits, decay_logits = self.gates_projection(x).chunk(2, dim=-1)
        beta = beta_logits.sigmoid()
        log_decay = -self.log_rates.float().exp() * F.softplus(
            decay_logits.float() + self.decay_bias
        )
        recurrent = state.recurrent if state is not None else None
        output, final_state = gated_delta_rule(
            query,
            key,
            value,
            log_decay,
            beta,
            initial_state=recurrent,
        )
        if state is not None:
            state.recurrent.copy_(final_state.detach())
        output_gate = output_gate.view(
            batch, length, self.spec.num_value_heads, self.spec.value_head_dim
        )
        if self.spec.output_gate_activation == "sigmoid":
            output_gate = output_gate.float().sigmoid()
        else:
            output_gate = F.silu(output_gate.float())
        output = self.output_norm(output) * output_gate.to(output.dtype)
        return self.output_projection(output.flatten(2))


class KimiDeltaAttention(nn.Module):
    """A fixed-state delta-rule mixer with an independent decay per key channel."""

    def __init__(self, hidden_size, spec, eps):
        super().__init__()
        self.spec = spec
        self.key_dim = spec.num_key_heads * spec.key_head_dim
        self.value_dim = spec.num_value_heads * spec.value_head_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.qkvz_projection = Linear(
            hidden_size,
            2 * self.key_dim + 2 * self.value_dim,
            bias=False,
        )
        self.beta_projection = Linear(hidden_size, spec.num_value_heads, bias=False)
        self.decay_down_projection = Linear(hidden_size, spec.value_head_dim, bias=False)
        self.decay_up_projection = Linear(
            spec.value_head_dim,
            spec.num_value_heads * spec.key_head_dim,
            bias=False,
        )
        self.conv_kernel = nn.Parameter(torch.empty(self.conv_dim, 1, spec.conv_kernel_size))
        self.log_rates = nn.Parameter(torch.zeros(spec.num_value_heads))
        self.decay_bias = nn.Parameter(torch.zeros(spec.num_value_heads * spec.key_head_dim))
        self.output_norm = RMSNorm(spec.value_head_dim, eps)
        self.output_projection = Linear(self.value_dim, hidden_size, bias=False)

    def forward(self, x, state=None):
        batch, length, _ = x.shape
        mixed = self.qkvz_projection(x)
        query, key, value, output_gate = torch.split(
            mixed,
            (self.key_dim, self.key_dim, self.value_dim, self.value_dim),
            dim=-1,
        )
        conv_input = torch.cat((query, key, value), dim=-1).transpose(1, 2)
        history = self.spec.conv_kernel_size - 1
        if state is not None:
            conv_input = torch.cat((state.convolution, conv_input), dim=2)
        convolved = F.silu(causal_depthwise_conv1d(conv_input, self.conv_kernel))
        if state is not None:
            state.convolution.copy_(conv_input[:, :, -history:].detach())
            convolved = convolved[:, :, history:]
        query, key, value = torch.split(
            convolved.transpose(1, 2),
            (self.key_dim, self.key_dim, self.value_dim),
            dim=-1,
        )
        query = query.view(
            batch,
            length,
            self.spec.num_key_heads,
            self.spec.key_head_dim,
        )
        key = key.view(
            batch,
            length,
            self.spec.num_key_heads,
            self.spec.key_head_dim,
        )
        value = value.view(
            batch,
            length,
            self.spec.num_value_heads,
            self.spec.value_head_dim,
        )
        beta = self.beta_projection(x).sigmoid()
        decay_logits = self.decay_up_projection(self.decay_down_projection(x)).view(
            batch,
            length,
            self.spec.num_value_heads,
            self.spec.key_head_dim,
        )
        recurrent = state.recurrent if state is not None else None
        output, final_state = kimi_delta_rule(
            query,
            key,
            value,
            decay_logits,
            beta,
            self.log_rates,
            self.decay_bias,
            initial_state=recurrent,
        )
        if state is not None:
            state.recurrent.copy_(final_state.detach())
        output_gate = output_gate.view(
            batch,
            length,
            self.spec.num_value_heads,
            self.spec.value_head_dim,
        )
        if self.spec.output_gate_activation == "sigmoid":
            output_gate = output_gate.float().sigmoid()
        else:
            output_gate = F.silu(output_gate.float())
        output = self.output_norm(output) * output_gate.to(output.dtype)
        return self.output_projection(output.flatten(2))


class SwiGLU(nn.Module):
    def __init__(self, hidden_size, spec):
        super().__init__()
        self.gate_proj = Linear(hidden_size, spec.intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, spec.intermediate_size, bias=False)
        self.down_proj = Linear(spec.intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


@dataclass(frozen=True)
class RoutingLayerStats:
    layer: str
    mean_probabilities: torch.Tensor
    utilization: torch.Tensor
    entropy: torch.Tensor
    load_balance_loss: torch.Tensor
    z_loss: torch.Tensor


@dataclass(frozen=True)
class CausalLMTrainingOutput:
    total_loss: torch.Tensor
    lm_loss: torch.Tensor
    load_balance_loss: torch.Tensor
    z_loss: torch.Tensor
    routing: tuple[RoutingLayerStats, ...]


def _reference_expert_swiglu(inputs, expert_ids, gate, up, down):
    """Execute sorted routes with a portable expert loop."""

    output = inputs.new_zeros((inputs.size(0), down.size(1)))
    for expert in range(gate.size(0)):
        positions = torch.nonzero(expert_ids == expert, as_tuple=False).flatten()
        selected = inputs.index_select(0, positions)
        hidden = F.silu(F.linear(selected, gate[expert].to(inputs.dtype)))
        hidden = hidden * F.linear(selected, up[expert].to(inputs.dtype))
        values = F.linear(hidden, down[expert].to(inputs.dtype))
        output.index_copy_(0, positions, values)
    return output


def _grouped_expert_swiglu(inputs, counts, gate, up, down):
    """Execute sorted BF16 routes through CUDA grouped matrix multiplications."""

    offsets = counts.cumsum(0).to(torch.int32)
    gate_values = torch._grouped_mm(inputs, gate.transpose(1, 2), offsets)
    up_values = torch._grouped_mm(inputs, up.transpose(1, 2), offsets)
    hidden = F.silu(gate_values) * up_values
    return torch._grouped_mm(hidden, down.transpose(1, 2), offsets)


class RoutedSwiGLU(nn.Module):
    """Token-choice dropless routed SwiGLU with contiguous expert banks."""

    def __init__(self, hidden_size, spec):
        super().__init__()
        self.spec = spec
        self.router = Linear(hidden_size, spec.num_experts, bias=False)
        self.gate_proj = nn.Parameter(
            torch.empty(spec.num_experts, spec.intermediate_size, hidden_size)
        )
        self.up_proj = nn.Parameter(
            torch.empty(spec.num_experts, spec.intermediate_size, hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(spec.num_experts, hidden_size, spec.intermediate_size)
        )

    def forward(self, x):
        shape = x.shape
        tokens = x.reshape(-1, shape[-1])
        logits = F.linear(tokens.float(), self.router.weight.float())
        probabilities = logits.softmax(dim=-1)
        selected_logits, selected_experts = logits.topk(self.spec.top_k, dim=-1)
        mixture = selected_logits.softmax(dim=-1)

        route_experts = selected_experts.flatten()
        route_tokens = (
            torch.arange(tokens.size(0), device=tokens.device)[:, None]
            .expand(-1, self.spec.top_k)
            .reshape(-1)
        )
        order = route_experts.argsort(stable=True)
        sorted_experts = route_experts.index_select(0, order)
        sorted_tokens = route_tokens.index_select(0, order)
        sorted_inputs = tokens.index_select(0, sorted_tokens)
        counts = torch.bincount(sorted_experts, minlength=self.spec.num_experts)

        grouped = (
            x.device.type == "cuda"
            and x.dtype == torch.bfloat16
            and torch.cuda.get_device_capability(x.device) >= (8, 0)
        )
        if grouped:
            routed = _grouped_expert_swiglu(
                sorted_inputs,
                counts,
                self.gate_proj.to(x.dtype),
                self.up_proj.to(x.dtype),
                self.down_proj.to(x.dtype),
            )
        else:
            routed = _reference_expert_swiglu(
                sorted_inputs,
                sorted_experts,
                self.gate_proj,
                self.up_proj,
                self.down_proj,
            )
        sorted_mixture = mixture.flatten().index_select(0, order).to(routed.dtype)
        combined = tokens.new_zeros(tokens.shape)
        combined.index_add_(0, sorted_tokens, routed * sorted_mixture[:, None])

        mean_probabilities = probabilities.mean(dim=0)
        utilization = counts.to(probabilities.dtype) / route_experts.numel()
        entropy = -(probabilities * probabilities.clamp_min(1e-20).log()).sum(dim=-1).mean()
        load_balance_loss = self.spec.num_experts * torch.sum(mean_probabilities * utilization)
        z_loss = logits.logsumexp(dim=-1).square().mean()
        stats = RoutingLayerStats(
            layer="",
            mean_probabilities=mean_probabilities,
            utilization=utilization,
            entropy=entropy,
            load_balance_loss=load_balance_loss,
            z_loss=z_loss,
        )
        return combined.view(shape), stats
