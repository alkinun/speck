"""compact speck language model."""

from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LayerConfig:
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_key_value_heads: int | None = 4

    def __post_init__(self):
        if self.hidden_size < 1 or self.intermediate_size < 1:
            raise ValueError("layer dimensions must be positive")
        if self.num_key_value_heads is not None and self.num_key_value_heads < 1:
            raise ValueError("kv heads must be positive")


def default_layers():
    return tuple(
        LayerConfig(num_key_value_heads=4 if index % 2 == 0 else None)
        for index in range(12)
    )


@dataclass(frozen=True)
class Config:
    vocab_size: int = 32000
    bos_token_id: int = 1
    eos_token_id: int = 2
    layers: tuple[LayerConfig, ...] = field(default_factory=default_layers)
    head_dim: int = 64
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    initializer_range: float = 0.02

    def __post_init__(self):
        layers = tuple(
            layer if isinstance(layer, LayerConfig) else LayerConfig(**layer)
            for layer in self.layers
        )
        object.__setattr__(self, "layers", layers)
        if not layers:
            raise ValueError("model must contain at least one layer")
        if self.head_dim < 2 or self.head_dim % 2:
            raise ValueError("head dimension must be positive and even")
        if self.max_position_embeddings < 1:
            raise ValueError("maximum position embeddings must be positive")
        for layer in layers:
            if layer.hidden_size % self.head_dim:
                raise ValueError("layer hidden size must be divisible by head dimension")
            query_heads = layer.hidden_size // self.head_dim
            if layer.num_key_value_heads is not None and query_heads % layer.num_key_value_heads:
                raise ValueError("query heads must be divisible by kv heads")

    @classmethod
    def from_dict(cls, settings):
        settings = dict(settings)
        settings.pop("architecture", None)
        settings.pop("expected_parameters", None)
        rope_parameters = settings.pop("rope_parameters", None)
        if rope_parameters and "rope_theta" not in settings:
            settings["rope_theta"] = rope_parameters.get("rope_theta", 10000.0)
        if "layers" in settings:
            layers = tuple(LayerConfig(**layer) for layer in settings.pop("layers"))
            for key in (
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "attention_every",
            ):
                settings.pop(key, None)
        else:
            hidden_size = settings.pop("hidden_size", 1024)
            intermediate_size = settings.pop("intermediate_size", 4096)
            num_hidden_layers = settings.pop("num_hidden_layers", 12)
            num_attention_heads = settings.pop("num_attention_heads", 16)
            num_key_value_heads = settings.pop("num_key_value_heads", 4)
            attention_every = settings.pop("attention_every", 2)
            head_dim = settings.get("head_dim", 64)
            if hidden_size != num_attention_heads * head_dim:
                raise ValueError("hidden size must equal attention heads times head dimension")
            if num_hidden_layers < 1 or attention_every < 1:
                raise ValueError("layer count and attention interval must be positive")
            layers = tuple(
                LayerConfig(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_key_value_heads=(
                        num_key_value_heads if index % attention_every == 0 else None
                    ),
                )
                for index in range(num_hidden_layers)
            )
        for key in (
            "architectures",
            "auto_map",
            "attention_bias",
            "attention_dropout",
            "dtype",
            "hidden_act",
            "mlp_bias",
            "model_type",
            "pad_token_id",
            "qk_norm",
            "tie_word_embeddings",
            "transformers_version",
            "use_cache",
        ):
            settings.pop(key, None)
        return cls(layers=layers, **settings)

    @property
    def num_attention_layers(self):
        return sum(layer.num_key_value_heads is not None for layer in self.layers)

    @property
    def embedding_size(self):
        return self.layers[0].hidden_size

    def settings(self):
        return {
            "vocab_size": self.vocab_size,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "layers": [asdict(layer) for layer in self.layers],
            "head_dim": self.head_dim,
            "max_position_embeddings": self.max_position_embeddings,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "initializer_range": self.initializer_range,
        }

    def export(self):
        return {
            "architectures": ["SpeckForCausalLM"],
            "auto_map": {
                "AutoConfig": "configuration_speck.SpeckConfig",
                "AutoModelForCausalLM": "modeling_speck.SpeckForCausalLM",
            },
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": self.bos_token_id,
            "dtype": "bfloat16",
            "eos_token_id": self.eos_token_id,
            "head_dim": self.head_dim,
            "hidden_act": "silu",
            "hidden_size": self.embedding_size,
            "initializer_range": self.initializer_range,
            "intermediate_size": self.layers[0].intermediate_size,
            "layers": [asdict(layer) for layer in self.layers],
            "max_position_embeddings": self.max_position_embeddings,
            "mlp_bias": False,
            "model_type": "speck",
            "num_attention_heads": self.embedding_size // self.head_dim,
            "num_hidden_layers": len(self.layers),
            "num_key_value_heads": next(
                (
                    layer.num_key_value_heads
                    for layer in self.layers
                    if layer.num_key_value_heads is not None
                ),
                1,
            ),
            "pad_token_id": None,
            "rms_norm_eps": self.rms_norm_eps,
            "qk_norm": True,
            "rope_theta": self.rope_theta,
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
        shapes = [
            (batch_size, layer.num_key_value_heads, length, config.head_dim)
            for layer in config.layers
            if layer.num_key_value_heads is not None
        ]
        self.keys = [torch.empty(shape, device=device, dtype=dtype) for shape in shapes]
        self.values = [torch.empty(shape, device=device, dtype=dtype) for shape in shapes]
        self.position = 0
        self.length = length

    def bytes_per_token(self):
        return sum(
            tensor.size(0) * tensor.size(1) * tensor.size(3) * tensor.element_size()
            for tensor in self.keys + self.values
        )


class Attention(nn.Module):
    def __init__(self, config, layer):
        super().__init__()
        q = layer.hidden_size
        kv = layer.num_key_value_heads * config.head_dim
        self.q_proj = Linear(layer.hidden_size, q, bias=False)
        self.k_proj = Linear(layer.hidden_size, kv, bias=False)
        self.v_proj = Linear(layer.hidden_size, kv, bias=False)
        self.o_proj = Linear(q, layer.hidden_size, bias=False)
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.q_heads = layer.hidden_size // config.head_dim
        self.kv_heads = layer.num_key_value_heads
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
    def __init__(self, layer):
        super().__init__()
        self.gate_proj = Linear(layer.hidden_size, layer.intermediate_size, bias=False)
        self.up_proj = Linear(layer.hidden_size, layer.intermediate_size, bias=False)
        self.down_proj = Linear(layer.intermediate_size, layer.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, config, layer, input_size, attention_index=None):
        super().__init__()
        self.attention_index = attention_index
        self.input_projection = (
            Linear(input_size, layer.hidden_size, bias=False)
            if input_size != layer.hidden_size
            else None
        )
        self.self_attn = Attention(config, layer) if attention_index is not None else None
        self.mlp = MLP(layer)
        if self.self_attn is not None:
            self.attention_norm = RMSNorm(layer.hidden_size, config.rms_norm_eps)
        self.mlp_norm = RMSNorm(layer.hidden_size, config.rms_norm_eps)

    def forward(self, x, cos, sin, cache=None):
        if self.input_projection is not None:
            x = self.input_projection(x)
        if self.self_attn is not None:
            x = x + self.self_attn(
                self.attention_norm(x), cos, sin, cache, self.attention_index
            )
        return x + self.mlp(self.mlp_norm(x))


class Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.embedding_size)
        attention_index = 0
        layers = []
        input_size = config.embedding_size
        for layer in config.layers:
            has_attention = layer.num_key_value_heads is not None
            layers.append(
                Block(
                    config,
                    layer,
                    input_size,
                    attention_index if has_attention else None,
                )
            )
            attention_index += has_attention
            input_size = layer.hidden_size
        self.layers = nn.ModuleList(layers)
        self.norm = RMSNorm(config.layers[-1].hidden_size, config.rms_norm_eps)
        self.output_projection = (
            Linear(config.layers[-1].hidden_size, config.embedding_size, bias=False)
            if config.layers[-1].hidden_size != config.embedding_size
            else None
        )


class SpeckForCausalLM(nn.Module):
    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(self, config=Config()):
        super().__init__()
        self.config = config
        self.model = Backbone(config)
        self.lm_head = Linear(config.embedding_size, config.vocab_size, bias=False)
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
        if self.model.output_projection is not None:
            x = self.model.output_projection(x)
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
        attention = 12 * sequence_length * sum(
            layer.hidden_size
            for layer in self.config.layers
            if layer.num_key_value_heads is not None
        )
        return 6 * linear + attention


def build_model(settings, vocab_size, bos_token_id=1, eos_token_id=2):
    settings = dict(settings)
    if settings.get("architecture_version") == 3:
        from speck.architecture import ArchitectureConfig
        from speck.model_v3 import SpeckV3ForCausalLM

        settings.update(
            vocab_size=vocab_size,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )
        config = ArchitectureConfig.from_dict(settings)
        model = SpeckV3ForCausalLM(config)
        if (
            config.expected_parameters is not None
            and model.parameter_count() != config.expected_parameters
        ):
            raise ValueError(f"unexpected parameter count: {model.parameter_count():,}")
        return model
    architecture = settings.pop("architecture", "speck")
    if architecture != "speck":
        raise ValueError(f"unsupported model architecture: {architecture}")
    expected_parameters = settings.pop("expected_parameters", None)
    settings.update(
        vocab_size=vocab_size,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
    )
    model = SpeckForCausalLM(Config.from_dict(settings))
    if expected_parameters is not None and model.parameter_count() != expected_parameters:
        raise ValueError(f"unexpected parameter count: {model.parameter_count():,}")
    return model
