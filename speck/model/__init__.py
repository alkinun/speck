"""Implement the Speck hybrid decoder language model."""

from collections import defaultdict
from dataclasses import replace

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from speck.model.architecture import (
    ArchitectureConfig,
    AttentionSpec,
    GatedCausalConvSpec,
    GatedDeltaNetSpec,
    KimiDeltaAttentionSpec,
    RoutedSwiGLUSpec,
    SwiGLUSpec,
)
from speck.model.layers import _LOSS_BACKENDS
from speck.model.layers import (
    Attention as Attention,
)
from speck.model.layers import (
    CausalLMTrainingOutput as CausalLMTrainingOutput,
)
from speck.model.layers import (
    GatedCausalConv as GatedCausalConv,
)
from speck.model.layers import (
    GatedDeltaNet as GatedDeltaNet,
)
from speck.model.layers import (
    KimiDeltaAttention as KimiDeltaAttention,
)
from speck.model.layers import (
    Linear as Linear,
)
from speck.model.layers import (
    RMSNorm as RMSNorm,
)
from speck.model.layers import (
    RotaryEmbedding as RotaryEmbedding,
)
from speck.model.layers import (
    RoutedSwiGLU as RoutedSwiGLU,
)
from speck.model.layers import (
    RoutingLayerStats as RoutingLayerStats,
)
from speck.model.layers import (
    SwiGLU as SwiGLU,
)
from speck.model.layers import (
    _grouped_expert_swiglu as _grouped_expert_swiglu,
)
from speck.model.layers import (
    _reference_expert_swiglu as _reference_expert_swiglu,
)
from speck.model.layers import (
    attention_rotary_key as attention_rotary_key,
)
from speck.model.layers import (
    causal_attention_mask as causal_attention_mask,
)
from speck.model.layers import (
    causal_depthwise_conv1d as causal_depthwise_conv1d,
)
from speck.model.layers import (
    flex_sliding_window_attention as flex_sliding_window_attention,
)
from speck.model.layers import (
    gated_delta_rule as gated_delta_rule,
)
from speck.model.layers import (
    initialize_delta_timescales as initialize_delta_timescales,
)
from speck.model.layers import (
    kimi_delta_rule as kimi_delta_rule,
)
from speck.model.layers import (
    liger_linear_cross_entropy as liger_linear_cross_entropy,
)
from speck.model.layers import (
    linear_cross_entropy as linear_cross_entropy,
)
from speck.model.layers import (
    mean_causal_attention_context as mean_causal_attention_context,
)
from speck.model.layers import (
    ordered_block_rows as ordered_block_rows,
)
from speck.model.layers import (
    rotate as rotate,
)
from speck.model.layers import (
    sliding_window_block_mask as sliding_window_block_mask,
)
from speck.model.layers import (
    torch_gated_delta_rule as torch_gated_delta_rule,
)
from speck.model.layers import (
    torch_kimi_delta_rule as torch_kimi_delta_rule,
)
from speck.model.layers import (
    torch_sliding_window_attention as torch_sliding_window_attention,
)
from speck.model.state import (
    AttentionState as AttentionState,
)
from speck.model.state import (
    ConvolutionState as ConvolutionState,
)
from speck.model.state import (
    DeltaNetState as DeltaNetState,
)
from speck.model.state import (
    SequenceState as SequenceState,
)
from speck.training.optimizers import (
    BatchedMuon as BatchedMuon,
)
from speck.training.optimizers import (
    CombinedOptimizer as CombinedOptimizer,
)
from speck.training.optimizers import (
    DeviceAdamW as DeviceAdamW,
)


class Operation(nn.Module):
    def __init__(self, hidden_size, spec, config):
        super().__init__()
        self.spec = spec
        self.norm = RMSNorm(hidden_size, config.rms_norm_eps)
        if isinstance(spec, AttentionSpec):
            self.operation = Attention(hidden_size, spec, config.rms_norm_eps)
        elif isinstance(spec, GatedCausalConvSpec):
            self.operation = GatedCausalConv(hidden_size, spec)
        elif isinstance(spec, GatedDeltaNetSpec):
            self.operation = GatedDeltaNet(hidden_size, spec, config.rms_norm_eps)
        elif isinstance(spec, KimiDeltaAttentionSpec):
            self.operation = KimiDeltaAttention(hidden_size, spec, config.rms_norm_eps)
        elif isinstance(spec, SwiGLUSpec):
            self.operation = SwiGLU(hidden_size, spec)
        elif isinstance(spec, RoutedSwiGLUSpec):
            self.operation = RoutedSwiGLU(hidden_size, spec)
        else:
            raise TypeError("unsupported architecture operation")

    def forward(self, x, rotary, position, state=None, memory=None, produced=None):
        normalized = self.norm(x)
        if isinstance(self.spec, AttentionSpec):
            rotary_dim = self.spec.active_rope_dim
            embedding = rotary[attention_rotary_key(self.spec)] if rotary_dim else None
            return (
                self.operation(normalized, embedding, position, state, memory, produced),
                None,
            )
        if isinstance(
            self.spec,
            (GatedCausalConvSpec, GatedDeltaNetSpec, KimiDeltaAttentionSpec),
        ):
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

    def forward(
        self,
        x,
        rotary,
        position,
        state,
        occurrence,
        memory=None,
        produced=None,
        masked_routed_layers=(),
    ):
        outputs = []
        routing = []
        for branch_index, branch in enumerate(self.branches):
            key = f"occurrence_{occurrence}_stage_{self.stage_index}_branch_{branch_index}"
            entry = state.entries[key] if state is not None and key in state.entries else None
            if key in masked_routed_layers and isinstance(branch.operation, RoutedSwiGLU):
                output, stats = torch.zeros_like(x), None
            else:
                output, stats = branch(x, rotary, position, entry, memory, produced)
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

    def forward(self, x, rotary, position, state, occurrence, memory=None, masked=()):
        produced = {}
        routing = []
        for stage in self.stages:
            x, stage_routing = stage(
                x,
                rotary,
                position,
                state,
                occurrence,
                memory,
                produced,
                masked,
            )
            routing.extend(stage_routing)
        return x, produced, tuple(routing)


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
        rotary_dimensions = {
            (
                branch.scope,
                branch.head_dim,
                branch.head_dim if branch.rope_dim is None else branch.rope_dim,
            )
            for invocation in self.execution_plan
            for stage in invocation.block.stages
            for branch in stage.branches
            if isinstance(branch, AttentionSpec)
            and (branch.head_dim if branch.rope_dim is None else branch.rope_dim) > 0
        }
        self.rotary = nn.ModuleDict(
            {
                f"{scope}:{head_dim}:{rotary_dim}": RotaryEmbedding(
                    rotary_dim,
                    config.rope_theta,
                    config.rope_scaling_factor if scope == "global" else 1.0,
                )
                for scope, head_dim, rotary_dim in rotary_dimensions
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
            elif isinstance(module, GatedCausalConv):
                nn.init.normal_(module.kernel, std=self.config.initializer_range)
            elif isinstance(module, RoutedSwiGLU):
                for bank in (module.gate_proj, module.up_proj, module.down_proj):
                    nn.init.normal_(bank, std=self.config.initializer_range)
            elif isinstance(module, GatedDeltaNet):
                nn.init.normal_(module.conv_kernel, std=self.config.initializer_range)
                if module.spec.decay_initialization == "fla":
                    initialize_delta_timescales(
                        module.log_rates,
                        module.decay_bias,
                        minimum_rate=0.0,
                    )
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
                        if branch.reads_memory:
                            continue
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
                            storage_dtype=kv_cache_dtype,
                        )
                    elif isinstance(branch, GatedCausalConvSpec):
                        entries[key] = ConvolutionState(
                            batch_size,
                            branch.inner_size,
                            branch.kernel_size - 1,
                            device,
                            dtype,
                        )
                    elif isinstance(branch, (GatedDeltaNetSpec, KimiDeltaAttentionSpec)):
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
        return_training_output=False,
        load_balance_coefficient=0.01,
        router_z_loss_coefficient=0.001,
        masked_routed_layers=(),
    ):
        if (tokens is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of tokens or inputs_embeds")
        if targets is not None and last_token_only:
            raise ValueError("last-token logits cannot be used with full-sequence targets")
        if return_training_output and targets is None:
            raise ValueError("training output requires targets")
        if load_balance_coefficient < 0 or router_z_loss_coefficient < 0:
            raise ValueError("routing loss coefficients must be non-negative")
        masked_routed_layers = frozenset(masked_routed_layers)
        if masked_routed_layers:
            unknown_masks = masked_routed_layers - self.routed_operations().keys()
            if unknown_masks:
                raise ValueError(f"unknown routed layer masks: {', '.join(sorted(unknown_masks))}")
        x = self.embed_tokens(tokens) if inputs_embeds is None else inputs_embeds
        x = x.to(torch.bfloat16 if x.is_cuda else self.embed_tokens.weight.dtype)
        length = x.size(1)
        position = state.position if state is not None else 0
        maximum = state.length if state is not None else self.config.max_position_embeddings
        if position + length > maximum:
            raise ValueError("sequence exceeds the available model state")
        memory = {}
        routing = []
        for invocation, adapter in zip(self.execution_plan, self.adapters):
            x = adapter(x)
            core = self.cores[invocation.weight_key]
            if self.training and self.gradient_checkpointing and state is None:
                x, produced, block_routing = activation_checkpoint(
                    core,
                    x,
                    self.rotary,
                    position,
                    None,
                    invocation.occurrence_index,
                    memory,
                    masked_routed_layers,
                    use_reentrant=False,
                )
            else:
                x, produced, block_routing = core(
                    x,
                    self.rotary,
                    position,
                    state,
                    invocation.occurrence_index,
                    memory,
                    masked_routed_layers,
                )
            routing.extend(block_routing)
            if produced:
                memory = {**memory, **produced}
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
                z_loss = torch.stack([item.z_loss for item in routing]).mean() if routing else zero
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

    def routed_operations(self):
        operations = {}
        for invocation in self.execution_plan:
            core = self.cores[invocation.weight_key]
            for stage_index, stage in enumerate(core.stages):
                for branch_index, branch in enumerate(stage.branches):
                    if isinstance(branch.operation, RoutedSwiGLU):
                        key = (
                            f"occurrence_{invocation.occurrence_index}_stage_{stage_index}"
                            f"_branch_{branch_index}"
                        )
                        operations[key] = branch.operation
        return operations

    def routing_config(self):
        values = []
        for layer, operation in self.routed_operations().items():
            values.append(
                {
                    "layer": layer,
                    "intermediate_size": operation.spec.intermediate_size,
                    "num_experts": operation.spec.num_experts,
                    "top_k": operation.spec.top_k,
                }
            )
        return values

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
        if isinstance(optimizer, CombinedOptimizer):
            required_muon = {
                id(parameter)
                for operation in self.routed_operations().values()
                for parameter in (
                    operation.router.weight,
                    operation.gate_proj,
                    operation.up_proj,
                    operation.down_proj,
                )
            }
            if any(memberships[identifier] != "muon" for identifier in required_muon):
                raise ValueError("routed router and expert parameters must use Muon")
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
                        linear += 2 * hidden_size * hidden_size
                        if not branch.reads_memory:
                            linear += 2 * hidden_size * kv_size
                        if branch.output_gate == "headwise":
                            linear += hidden_size * (hidden_size // branch.head_dim)
                        elif branch.output_gate == "elementwise":
                            linear += hidden_size * hidden_size
                        window_size = None
                        if branch.scope == "sliding":
                            assert branch.window_size is not None
                            window_size = branch.window_size
                        context = mean_causal_attention_context(sequence_length, window_size)
                        attention += 12 * context * hidden_size
                    elif isinstance(branch, GatedCausalConvSpec):
                        linear += 4 * hidden_size * branch.inner_size
                        linear += branch.inner_size * branch.kernel_size
                    elif isinstance(branch, GatedDeltaNetSpec):
                        key_size = branch.num_key_heads * branch.key_head_dim
                        value_size = branch.num_value_heads * branch.value_head_dim
                        linear += hidden_size * (2 * key_size + 3 * value_size)
                        linear += 2 * hidden_size * branch.num_value_heads
                        linear += (2 * key_size + value_size) * branch.conv_kernel_size
                        attention += (
                            21
                            * branch.num_value_heads
                            * branch.key_head_dim
                            * branch.value_head_dim
                        )
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
                    elif isinstance(branch, RoutedSwiGLUSpec):
                        linear += branch.num_experts * hidden_size
                        linear += 3 * branch.top_k * hidden_size * branch.intermediate_size
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
