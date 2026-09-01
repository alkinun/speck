"""Implement the Speck hybrid decoder language model."""

import math
from collections import defaultdict
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from speck.architecture import (
    ArchitectureConfig,
    AttentionSpec,
    GatedCausalConvSpec,
    RoutedSwiGLUSpec,
    SwiGLUSpec,
)

_LOSS_BACKENDS = {"torch", "liger"}


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


class CombinedOptimizer:
    """Present multiple optimizers through one optimizer-like interface."""

    def __init__(self, **optimizers):
        self.optimizers = optimizers

    @property
    def param_groups(self):
        return [group for optimizer in self.optimizers.values() for group in optimizer.param_groups]

    def zero_grad(self, set_to_none=True):
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        for optimizer in self.optimizers.values():
            optimizer.step()

    def state_dict(self):
        return {
            "format_version": 1,
            "optimizers": {
                name: optimizer.state_dict() for name, optimizer in self.optimizers.items()
            },
        }

    def load_state_dict(self, state):
        if state.get("format_version") != 1:
            raise ValueError("unsupported combined optimizer state")
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state["optimizers"][name])

    def compile_step(self):
        """Compile the matrix-heavy Muon update with checkpoint-compatible state."""

        muon = self.optimizers.get("muon")
        if isinstance(muon, BatchedMuon):
            for group in muon.param_groups:
                if not isinstance(group["lr"], torch.Tensor):
                    parameter = group["params"][0]
                    group["lr"] = torch.tensor(group["lr"], device=parameter.device)
            muon.step = torch.compile(
                muon.step,
                dynamic=False,
                options={
                    "max_autotune": True,
                    "coordinate_descent_tuning": True,
                    "aggressive_fusion": True,
                },
            )


class BatchedMuon(torch.optim.Muon):
    """Run Muon's per-matrix Newton-Schulz updates in shape batches.

    Expert banks retain one checkpoint state entry while each leading-dimension
    matrix slice receives its own orthogonalized update.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        weight_decay=0.1,
        momentum=0.95,
        nesterov=True,
        ns_coefficients=(3.4445, -4.775, 2.0315),
        eps=1e-7,
        ns_steps=5,
        adjust_lr_fn=None,
    ):
        if isinstance(lr, torch.Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0 <= lr:
            raise ValueError("learning rate must be non-negative")
        if not 0 <= momentum:
            raise ValueError("momentum must be non-negative")
        if not 0 <= weight_decay:
            raise ValueError("weight decay must be non-negative")
        if adjust_lr_fn not in {None, "original", "match_rms_adamw"}:
            raise ValueError(f"unsupported Muon learning-rate adjustment: {adjust_lr_fn}")
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_coefficients": ns_coefficients,
            "eps": eps,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
        }
        torch.optim.Optimizer.__init__(self, params, defaults)
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.ndim not in (2, 3):
                    raise ValueError(
                        "BatchedMuon only supports matrices and expert matrix banks, "
                        f"got {tuple(parameter.shape)}"
                    )

    @staticmethod
    def _lr_ratio(mode, shape):
        rows, columns = shape
        if mode is None or mode == "original":
            return math.sqrt(max(1, rows / columns))
        if mode == "match_rms_adamw":
            return 0.2 * math.sqrt(max(rows, columns))
        return 1.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            batches = defaultdict(list)
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse or gradient.ndim not in (2, 3) or torch.is_complex(
                    parameter
                ):
                    raise RuntimeError("BatchedMuon requires dense, real matrix gradients or banks")
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        gradient, memory_format=torch.preserve_format
                    )
                parameters = parameter.unbind() if parameter.ndim == 3 else (parameter,)
                gradients = gradient.unbind() if gradient.ndim == 3 else (gradient,)
                momenta = (
                    state["momentum_buffer"].unbind()
                    if state["momentum_buffer"].ndim == 3
                    else (state["momentum_buffer"],)
                )
                for matrix, matrix_gradient, momentum in zip(
                    parameters, gradients, momenta
                ):
                    shape = tuple(matrix.shape)
                    lr_ratio = self._lr_ratio(group["adjust_lr_fn"], shape)
                    oriented_shape = (min(shape), max(shape))
                    batches[(oriented_shape, lr_ratio)].append(
                        (matrix, matrix_gradient, momentum, shape[0] > shape[1])
                    )

            for (_, lr_ratio), entries in batches.items():
                parameters, gradients, momentum_buffers, transposed = map(list, zip(*entries))
                torch._foreach_lerp_(momentum_buffers, gradients, 1 - group["momentum"])
                if group["nesterov"]:
                    updates = torch._foreach_lerp(gradients, momentum_buffers, group["momentum"])
                else:
                    updates = momentum_buffers
                orthogonal = torch.stack(
                    [
                        (update.T if transpose else update).bfloat16()
                        for update, transpose in zip(updates, transposed)
                    ]
                )
                norms = torch.linalg.vector_norm(orthogonal, dim=(1, 2), keepdim=True)
                orthogonal.div_(norms.clamp(min=group["eps"]))
                a, b, c = group["ns_coefficients"]
                for _ in range(group["ns_steps"]):
                    gram = torch.bmm(orthogonal, orthogonal.transpose(1, 2))
                    gram_update = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
                    orthogonal = torch.baddbmm(orthogonal, gram_update, orthogonal, beta=a)
                torch._foreach_mul_(parameters, 1 - group["lr"] * group["weight_decay"])
                final_updates = [
                    update.T if transpose else update
                    for update, transpose in zip(orthogonal.unbind(), transposed)
                ]
                adjusted_lr = group["lr"] * lr_ratio
                for parameter, update in zip(parameters, final_updates):
                    parameter.add_(update.to(parameter.dtype) * (-adjusted_lr))
        return loss


class DeviceAdamW(torch.optim.AdamW):
    """Keep fused AdamW enabled after loading legacy optimizer state."""

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        parameter = next(parameter for group in self.param_groups for parameter in group["params"])
        fused = parameter.device.type == "cuda"
        for group in self.param_groups:
            group["fused"] = fused


def rotate(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


def causal_depthwise_conv1d(x, weight):
    """Apply a tiny causal depthwise stencil without a generic convolution launch."""

    kernel = weight[:, 0].to(x.dtype)
    output = x * kernel[:, -1, None]
    for delay in range(1, kernel.size(1)):
        shifted = F.pad(x, (delay, 0))[:, :, :-delay]
        output = output + shifted * kernel[:, -1 - delay, None]
    return output


class AttentionState:
    """Maintain a bounded key-value cache in chronological ring-buffer order."""

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
            return self.keys[:, :, : self.used], self.values[:, :, : self.used]
        if self.write_position == 0:
            return self.keys, self.values
        return (
            torch.cat(
                (self.keys[:, :, self.write_position :], self.keys[:, :, : self.write_position]),
                dim=2,
            ),
            torch.cat(
                (
                    self.values[:, :, self.write_position :],
                    self.values[:, :, : self.write_position],
                ),
                dim=2,
            ),
        )

    def append(self, keys, values):
        length = keys.size(2)
        if length >= self.capacity:
            self.keys.copy_(keys[:, :, -self.capacity :])
            self.values.copy_(values[:, :, -self.capacity :])
            self.used = self.capacity
            self.write_position = 0
            return
        first = min(length, self.capacity - self.write_position)
        end = self.write_position + first
        self.keys[:, :, self.write_position : end] = keys[:, :, :first]
        self.values[:, :, self.write_position : end] = values[:, :, :first]
        remaining = length - first
        if remaining:
            self.keys[:, :, :remaining] = keys[:, :, first:]
            self.values[:, :, :remaining] = values[:, :, first:]
        self.write_position = (self.write_position + length) % self.capacity
        self.used = min(self.capacity, self.used + length)

    def allocated_bytes(self):
        return sum(tensor.numel() * tensor.element_size() for tensor in (self.keys, self.values))


class ConvolutionState:
    """Hold causal convolution history for incremental decoding."""

    def __init__(self, batch_size, inner_size, history, device, dtype):
        self.values = torch.zeros(batch_size, inner_size, history, device=device, dtype=dtype)

    def allocated_bytes(self):
        return self.values.numel() * self.values.element_size()


class SequenceState:
    """Track incremental-decoding position and per-operation caches."""

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
        frequency = 1 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        angles = torch.outer(torch.arange(length, dtype=torch.float32), frequency).repeat(1, 2)
        self.register_buffer("cos", angles.cos()[None, None], persistent=False)
        self.register_buffer("sin", angles.sin()[None, None], persistent=False)

    def forward(self, position, length, dtype):
        end = position + length
        cos = self.get_buffer("cos")
        sin = self.get_buffer("sin")
        return cos[..., position:end, :].to(dtype), sin[..., position:end, :].to(dtype)


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
        q = self.q_proj(x).view(batch, length, self.q_heads, self.spec.head_dim).transpose(1, 2)
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
        cos, sin = rotary(position, length, q.dtype)
        q = rotate(self.q_norm(q), cos, sin)
        k = rotate(self.k_norm(k), cos, sin)
        if state is None and self.spec.scope == "global":
            keys, values, mask, causal = k, v, None, True
        else:
            past_k, past_v = (k[:, :, :0], v[:, :, :0]) if state is None else state.current()
            keys = torch.cat((past_k, k), dim=2)
            values = torch.cat((past_v, v), dim=2)
            key_positions = torch.arange(
                position - past_k.size(2), position + length, device=x.device
            )
            query_positions = torch.arange(position, position + length, device=x.device)[:, None]
            mask = key_positions[None, :] <= query_positions
            if self.spec.scope == "sliding":
                window = self.spec.window_size
                assert window is not None
                mask &= key_positions[None, :] > query_positions - window
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
        transposed = (first_gate * values).transpose(1, 2)
        history = self.spec.kernel_size - 1
        if state is not None:
            transposed = torch.cat((state.values, transposed), dim=2)
        convolved = causal_depthwise_conv1d(transposed, self.kernel)
        if state is not None:
            state.values.copy_(transposed[:, :, -history:])
            convolved = convolved[:, :, history:]
        return self.output_projection(second_gate * convolved.transpose(1, 2))


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
        load_balance_loss = self.spec.num_experts * torch.sum(
            mean_probabilities * utilization
        )
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
        elif isinstance(spec, RoutedSwiGLUSpec):
            self.operation = RoutedSwiGLU(hidden_size, spec)
        else:
            raise TypeError("unsupported architecture operation")

    def forward(self, x, rotary, position, state=None):
        normalized = self.norm(x)
        if isinstance(self.spec, AttentionSpec):
            return (
                self.operation(
                    normalized, rotary[str(self.spec.head_dim)], position, state
                ),
                None,
            )
        if isinstance(self.spec, GatedCausalConvSpec):
            return self.operation(normalized, state), None
        if isinstance(self.spec, RoutedSwiGLUSpec):
            return self.operation(normalized)
        return self.operation(normalized), None


class Stage(nn.Module):
    def __init__(self, hidden_size, config, stage_index, stage):
        super().__init__()
        self.stage_index = stage_index
        self.branches = nn.ModuleList(
            Operation(hidden_size, spec, config) for spec in stage.branches
        )

    def forward(self, x, rotary, position, state, occurrence):
        outputs = []
        routing = []
        for branch_index, branch in enumerate(self.branches):
            key = f"occurrence_{occurrence}_stage_{self.stage_index}_branch_{branch_index}"
            entry = state.entries[key] if state is not None and key in state.entries else None
            output, stats = branch(x, rotary, position, entry)
            outputs.append(output)
            if stats is not None:
                routing.append(replace(stats, layer=key))
        return x + sum(outputs), tuple(routing)


class BlockCore(nn.Module):
    def __init__(self, block, config):
        super().__init__()
        self.stages = nn.ModuleList(
            Stage(block.hidden_size, config, index, stage)
            for index, stage in enumerate(block.stages)
        )

    def forward(self, x, rotary, position, state, occurrence):
        routing = []
        for stage in self.stages:
            x, stage_routing = stage(x, rotary, position, state, occurrence)
            routing.extend(stage_routing)
        return x, tuple(routing)


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
                    head_dim, config.max_position_embeddings, config.rope_theta
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
            elif isinstance(module, RoutedSwiGLU):
                for bank in (module.gate_proj, module.up_proj, module.down_proj):
                    nn.init.normal_(bank, std=self.config.initializer_range)

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
        )
        return self.embed_tokens

    def state(self, batch_size=1, length=None, device=None, dtype=None):
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
                        if branch.scope == "global":
                            capacity = length
                        else:
                            assert branch.window_size is not None
                            capacity = min(length, branch.window_size)
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
        loss_reduction="mean",
        return_training_output=False,
        load_balance_coefficient=0.01,
        router_z_loss_coefficient=0.001,
    ):
        if (tokens is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of tokens or inputs_embeds")
        if targets is not None and last_token_only:
            raise ValueError("last-token logits cannot be used with full-sequence targets")
        if return_training_output and targets is None:
            raise ValueError("training output requires targets")
        if load_balance_coefficient < 0 or router_z_loss_coefficient < 0:
            raise ValueError("routing loss coefficients must be non-negative")
        x = self.embed_tokens(tokens) if inputs_embeds is None else inputs_embeds
        x = x.to(torch.bfloat16 if x.is_cuda else self.embed_tokens.weight.dtype)
        length = x.size(1)
        position = state.position if state is not None else 0
        maximum = state.length if state is not None else self.config.max_position_embeddings
        if position + length > maximum:
            raise ValueError("sequence exceeds the available model state")
        routing = []
        for invocation, adapter in zip(self.execution_plan, self.adapters):
            x = adapter(x)
            x, block_routing = self.cores[invocation.weight_key](
                x, self.rotary, position, state, invocation.occurrence_index
            )
            routing.extend(block_routing)
        if state is not None:
            state.position += length
        hidden = self.output_projection(self.norm(x))
        if targets is not None:
            lm_loss = linear_cross_entropy(
                hidden,
                self.lm_head.weight,
                targets,
                loss_reduction,
                self.loss_backend,
            )
            if return_training_output:
                zero = lm_loss.new_zeros(())
                load_balance_loss = (
                    torch.stack([item.load_balance_loss for item in routing]).mean()
                    if routing
                    else zero
                )
                z_loss = (
                    torch.stack([item.z_loss for item in routing]).mean()
                    if routing
                    else zero
                )
                output = CausalLMTrainingOutput(
                    total_loss=lm_loss
                    + load_balance_coefficient * load_balance_loss
                    + router_z_loss_coefficient * z_loss,
                    lm_loss=lm_loss,
                    load_balance_loss=load_balance_loss,
                    z_loss=z_loss,
                    routing=tuple(routing),
                )
            else:
                output = lm_loss
        else:
            output = self.lm_head(hidden[:, -1:] if last_token_only else hidden).float()
        return (output, hidden) if return_hidden else output

    def optimizer(self, lr=6e-4, weight_decay=0.1, name="adamw"):
        embedding = self.embed_tokens.weight
        expert_banks = {
            id(parameter)
            for module in self.modules()
            if isinstance(module, RoutedSwiGLU)
            for parameter in (module.gate_proj, module.up_proj, module.down_proj)
        }
        matrices, other_decay, no_decay = [], [], []
        for parameter in self.parameters():
            if parameter is embedding or parameter.ndim < 2:
                no_decay.append(parameter)
            elif parameter.ndim == 2 or id(parameter) in expert_banks:
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
                        if branch.scope == "global":
                            context = sequence_length
                        else:
                            assert branch.window_size is not None
                            context = min(sequence_length, branch.window_size)
                        attention += 12 * context * hidden_size
                    elif isinstance(branch, GatedCausalConvSpec):
                        linear += 4 * hidden_size * branch.inner_size
                        linear += branch.inner_size * branch.kernel_size
                    elif isinstance(branch, RoutedSwiGLUSpec):
                        linear += branch.num_experts * hidden_size
                        linear += 3 * branch.top_k * hidden_size * branch.intermediate_size
                    else:
                        linear += 3 * hidden_size * branch.intermediate_size
            input_size = hidden_size
        if input_size != self.config.embedding_size:
            linear += input_size * self.config.embedding_size
        return 6 * linear + attention


def build_model(settings, vocab_size, bos_token_id=1, eos_token_id=2, loss_backend="torch"):
    values = dict(settings)
    values.update(
        vocab_size=vocab_size,
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
