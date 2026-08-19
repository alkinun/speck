"""compact llama model."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Config:
    vocab_size: int = 32000
    bos_token_id: int = 1
    eos_token_id: int = 2
    hidden_size: int = 384
    intermediate_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 6
    num_key_value_heads: int = 2
    head_dim: int = 64
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    initializer_range: float = 0.02

    def __post_init__(self):
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden size must equal attention heads times head dimension")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("query heads must be divisible by kv heads")

    def export(self):
        return {
            "architectures": ["LlamaForCausalLM"],
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
            "model_type": "llama",
            "num_attention_heads": self.num_attention_heads,
            "num_hidden_layers": self.num_hidden_layers,
            "num_key_value_heads": self.num_key_value_heads,
            "pad_token_id": None,
            "pretraining_tp": 1,
            "rms_norm_eps": self.rms_norm_eps,
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


def rotate(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


class KVCache:
    def __init__(self, config, batch_size, length, device, dtype):
        shape = (batch_size, config.num_key_value_heads, length, config.head_dim)
        self.keys = [torch.empty(shape, device=device, dtype=dtype) for _ in range(config.num_hidden_layers)]
        self.values = [torch.empty(shape, device=device, dtype=dtype) for _ in range(config.num_hidden_layers)]
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
        self.q_heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

    def forward(self, x, cos, sin, cache=None, layer_index=None):
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        causal = True
        mask = None
        if cache is not None:
            end = cache.position + length
            if end > cache.length:
                raise ValueError("kv cache is full")
            cache.keys[layer_index][:, :, cache.position:end] = k
            cache.values[layer_index][:, :, cache.position:end] = v
            k = cache.keys[layer_index][:, :, :end]
            v = cache.values[layer_index][:, :, :end]
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
    def __init__(self, config):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, x, cos, sin, cache=None, layer_index=None):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, cache, layer_index)
        return x + self.mlp(self.post_attention_layernorm(x))


class Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)


class Llama(nn.Module):
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
        for layer_index, layer in enumerate(self.model.layers):
            x = layer(x, cos, sin, cache, layer_index)
        if cache is not None:
            cache.position += length
        logits = self.lm_head(self.model.norm(x)).float()
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

    def optimizer(self, lr=6e-4, weight_decay=0.1):
        embedding = self.model.embed_tokens.weight
        decay, no_decay = [], []
        for parameter in self.parameters():
            (no_decay if parameter is embedding or parameter.ndim < 2 else decay).append(parameter)
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
        attention = 12 * self.config.num_hidden_layers * self.config.hidden_size * sequence_length
        return 6 * linear + attention


def build_model(settings, vocab_size, bos_token_id=1, eos_token_id=2):
    settings = dict(settings)
    architecture = settings.pop("architecture", "llama")
    if architecture != "llama":
        raise ValueError(f"unsupported model architecture: {architecture}")
    expected_parameters = settings.pop("expected_parameters", None)
    model = Llama(Config(
        vocab_size=vocab_size,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        **settings,
    ))
    if expected_parameters is not None and model.parameter_count() != expected_parameters:
        raise ValueError(f"unexpected parameter count: {model.parameter_count():,}")
    return model
