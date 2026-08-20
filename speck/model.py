"""compact speck language model."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Config:
    vocab_size: int = 32000
    bos_token_id: int = 1
    eos_token_id: int = 2
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 12
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 64
    attention_every: int = 2
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    initializer_range: float = 0.02

    def __post_init__(self):
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden size must equal attention heads times head dimension")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("query heads must be divisible by kv heads")
        if self.attention_every < 1:
            raise ValueError("attention every must be positive")

    @property
    def num_attention_layers(self):
        return len(range(0, self.num_hidden_layers, self.attention_every))

    def export(self):
        return {
            "architectures": ["SpeckForCausalLM"],
            "auto_map": {
                "AutoConfig": "configuration_speck.SpeckConfig",
                "AutoModelForCausalLM": "modeling_speck.SpeckForCausalLM",
            },
            "attention_every": self.attention_every,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": self.bos_token_id,
            "dtype": "bfloat16",
            "eos_token_id": self.eos_token_id,
            "head_dim": self.head_dim,
            "hidden_act": "silu",
            "hidden_size": self.hidden_size,
            "initializer_range": self.initializer_range,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "mlp_bias": False,
            "model_type": "speck",
            "num_attention_heads": self.num_attention_heads,
            "num_hidden_layers": self.num_hidden_layers,
            "num_key_value_heads": self.num_key_value_heads,
            "pad_token_id": None,
            "rms_norm_eps": self.rms_norm_eps,
            "qk_norm": True,
            "rope_parameters": {"rope_theta": self.rope_theta, "rope_type": "default"},
            "tie_word_embeddings": True,
            "transformers_version": "5.14.1",
            "use_cache": True,
            "vocab_size": self.vocab_size,
        }


class Linear(nn.Linear):
    def forward(self, input):
        return F.linear(input, self.weight.to(input.dtype))


class RMSNorm(nn.Module):
    def __init__(self, size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x.float(), (x.size(-1),), eps=self.eps).to(x.dtype) * self.weight.to(x.dtype)


class CombinedOptimizer:
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
            "optimizers": {name: optimizer.state_dict() for name, optimizer in self.optimizers.items()},
        }

    def load_state_dict(self, state):
        if state.get("format_version") != 1:
            raise ValueError("unsupported combined optimizer state")
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state["optimizers"][name])


def rotate(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


class KVCache:
    def __init__(self, config, batch_size, length, device, dtype):
        shape = (batch_size, config.num_key_value_heads, length, config.head_dim)
        self.keys = [torch.empty(shape, device=device, dtype=dtype) for _ in range(config.num_attention_layers)]
        self.values = [torch.empty(shape, device=device, dtype=dtype) for _ in range(config.num_attention_layers)]
        self.position = 0
        self.length = length


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        q = config.num_attention_heads * config.head_dim
        kv = config.num_key_value_heads * config.head_dim
        self.q_proj = Linear(config.hidden_size, q, bias=False)
        self.k_proj = Linear(config.hidden_size, kv, bias=False)
        self.v_proj = Linear(config.hidden_size, kv, bias=False)
        self.o_proj = Linear(q, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.q_heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

    def forward(self, x, cos, sin, cache=None, attention_index=None):
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        q = rotate(self.q_norm(q), cos, sin)
        k = rotate(self.k_norm(k), cos, sin)
        causal = True
        mask = None
        if cache is not None:
            end = cache.position + length
            if end > cache.length:
                raise ValueError("kv cache is full")
            cache.keys[attention_index][:, :, cache.position:end] = k
            cache.values[attention_index][:, :, cache.position:end] = v
            k = cache.keys[attention_index][:, :, :end]
            v = cache.values[attention_index][:, :, :end]
            if length == 1:
                causal = False
            elif cache.position:
                rows = cache.position + torch.arange(length, device=x.device)[:, None]
                columns = torch.arange(end, device=x.device)[None, :]
                mask = columns <= rows
                causal = False
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=causal, enable_gqa=self.q_heads != self.kv_heads
        )
        return self.o_proj(y.transpose(1, 2).contiguous().view(batch, length, -1))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, config, attention_index=None):
        super().__init__()
        self.attention_index = attention_index
        self.self_attn = Attention(config) if attention_index is not None else None
        self.mlp = MLP(config)
        if self.self_attn is not None:
            self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, x, cos, sin, cache=None):
        if self.self_attn is not None:
            x = x + self.self_attn(
                self.attention_norm(x), cos, sin, cache, self.attention_index
            )
        return x + self.mlp(self.mlp_norm(x))


class Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        attention_index = 0
        layers = []
        for layer_index in range(config.num_hidden_layers):
            has_attention = layer_index % config.attention_every == 0
            layers.append(Block(config, attention_index if has_attention else None))
            attention_index += has_attention
        self.layers = nn.ModuleList(layers)
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)


class SpeckForCausalLM(nn.Module):
    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(self, config=Config()):
        super().__init__()
        self.config = config
        self.model = Backbone(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight
        frequency = 1 / (
            config.rope_theta
            ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim)
        )
        positions = torch.arange(config.max_position_embeddings, dtype=torch.float32)
        angles = torch.outer(positions, frequency).repeat(1, 2)
        self.register_buffer("cos", angles.cos()[None, None], persistent=False)
        self.register_buffer("sin", angles.sin()[None, None], persistent=False)

    @torch.no_grad()
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, (Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=self.config.initializer_range)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

    def forward(self, tokens, targets=None, cache=None):
        length = tokens.size(1)
        position = cache.position if cache is not None else 0
        if position + length > self.config.max_position_embeddings:
            raise ValueError("sequence exceeds max_position_embeddings")
        x = self.model.embed_tokens(tokens).to(torch.bfloat16 if tokens.is_cuda else torch.float32)
        cos = self.cos[..., position:position + length, :].to(x.dtype)
        sin = self.sin[..., position:position + length, :].to(x.dtype)
        for layer in self.model.layers:
            x = layer(x, cos, sin, cache)
        if cache is not None:
            cache.position += length
        x = self.model.norm(x)
        logits = self.lm_head(x).float()
        if targets is None:
            return logits
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten())

    def cache(self, batch_size=1, length=None):
        parameter = next(self.parameters())
        dtype = torch.bfloat16 if parameter.is_cuda else torch.float32
        return KVCache(
            self.config,
            batch_size,
            length or self.config.max_position_embeddings,
            parameter.device,
            dtype,
        )

    def optimizer(self, lr=6e-4, weight_decay=0.1, name="adamw"):
        embedding = self.model.embed_tokens.weight
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
            [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def flops_per_token(self, sequence_length):
        linear = sum(module.weight.numel() for module in self.modules() if isinstance(module, Linear))
        attention = 12 * self.config.num_attention_layers * self.config.hidden_size * sequence_length
        return 6 * linear + attention


def build_model(settings, vocab_size, bos_token_id=1, eos_token_id=2):
    settings = dict(settings)
    architecture = settings.pop("architecture", "speck")
    if architecture != "speck":
        raise ValueError(f"unsupported model architecture: {architecture}")
    expected_parameters = settings.pop("expected_parameters", None)
    model = SpeckForCausalLM(Config(
        vocab_size=vocab_size,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        **settings,
    ))
    if expected_parameters is not None and model.parameter_count() != expected_parameters:
        raise ValueError(f"unexpected parameter count: {model.parameter_count():,}")
    return model
