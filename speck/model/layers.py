"""Implement the Speck hybrid decoder language model."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right

_LOSS_BACKENDS = {"torch", "liger", "liger_aligned"}
_VOCAB_ALIGNMENT = 64


@torch.compiler.disable
def liger_linear_cross_entropy(hidden, weight, targets, reduction, aligned=False):
    try:
        from liger_kernel.transformers.functional import liger_fused_linear_cross_entropy
    except ImportError as exception:
        raise RuntimeError(
            "the Liger loss backend requires the GPU dependencies; run `uv sync --extra gpu`"
        ) from exception
    if not aligned:
        return liger_fused_linear_cross_entropy(hidden, weight, targets, reduction=reduction)
    # Pad the vocabulary to a multiple of 64 so the head GEMMs use aligned tensor-core kernels;
    # a -inf bias gives padded rows zero probability, leaving the loss and gradients unchanged.
    # The chunked weight gradient, the whole embedding gradient with tied embeddings,
    # accumulates in FP32.
    vocab = weight.size(0)
    padding = -vocab % _VOCAB_ALIGNMENT
    bias = None
    if padding:
        weight = F.pad(weight, (0, 0, 0, padding))
        bias = torch.zeros(vocab + padding, dtype=weight.dtype, device=weight.device)
        bias[vocab:] = float("-inf")
    return liger_fused_linear_cross_entropy(
        hidden,
        weight,
        targets,
        bias=bias,
        reduction=reduction,
        accum_dtype=torch.float32,
    )


def linear_cross_entropy(hidden, weight, targets, reduction, backend):
    hidden = hidden.flatten(0, 1)
    targets = targets.flatten()
    compute_weight = weight.to(hidden.dtype)
    if backend == "torch":
        logits = F.linear(hidden, compute_weight).float()
        return F.cross_entropy(logits, targets, reduction=reduction)
    return liger_linear_cross_entropy(
        hidden, compute_weight, targets, reduction, aligned=backend == "liger_aligned"
    )


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
    """Apply a tiny causal depthwise stencil without a generic convolution launch.

    One zero-padded buffer is built and every delayed tap is a view into it, so a
    kernel of width k costs a single allocation rather than one per tap.
    """

    kernel = weight[:, 0].to(x.dtype)
    taps = kernel.size(1)
    output = x * kernel[:, -1, None]
    if taps > 1:
        length = x.size(-1)
        padded = F.pad(x, (taps - 1, 0))
        for delay in range(1, taps):
            start = taps - 1 - delay
            shifted = padded.narrow(-1, start, length)
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
    return f"{spec.head_dim}:{spec.active_rope_dim}"


class Attention(nn.Module):
    """Global causal grouped-query attention with optional partial RoPE."""

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
        head_dim, kv_heads = self.spec.head_dim, self.spec.num_key_value_heads
        q = self.q_norm(self.q_proj(x).view(batch, length, self.q_heads, head_dim).transpose(1, 2))
        k = self.k_norm(self.k_proj(x).view(batch, length, kv_heads, head_dim).transpose(1, 2))
        v = self.v_proj(x).view(batch, length, kv_heads, head_dim).transpose(1, 2)
        rotary_dim = self.spec.active_rope_dim
        if rotary_dim:
            cos, sin = rotary(position, length, q.dtype)
            q = rotate(q, cos, sin, rotary_dim)
            k = rotate(k, cos, sin, rotary_dim)
        keys, values, past_length = k, v, 0
        if state is not None:
            past_k, past_v = state.current()
            past_length = past_k.size(2)
            if past_length:
                keys = torch.cat((past_k, k), dim=2)
                values = torch.cat((past_v, v), dim=2)
        if past_length == 0:
            mask, causal = None, True
        elif length == 1:
            mask, causal = None, False
        else:
            mask, causal = causal_lower_right(length, keys.size(2)), False
        output = F.scaled_dot_product_attention(
            q,
            keys,
            values,
            attn_mask=mask,
            is_causal=causal,
            enable_gqa=self.q_heads != kv_heads,
        )
        if state is not None:
            state.append(k, v)
        return self.o_proj(output.transpose(1, 2).contiguous().view(batch, length, hidden_size))


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
