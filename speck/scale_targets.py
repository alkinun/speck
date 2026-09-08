"""Generate and independently account non-launchable flagship scale targets."""

import hashlib
import json
from pathlib import Path

import torch

from speck.model import build_model

FORMAT = "speck_flagship_scale_target_accounting"
LENGTHS = (4_096, 131_072)


def _block(target, mixer):
    hidden = target["hidden_size"]
    if mixer == "attention":
        operation = {
            "kind": "attention",
            "head_dim": target["head_dim"],
            "num_key_value_heads": target["num_key_value_heads"],
            "rope_dim": 0,
        }
    else:
        operation = {
            "kind": "kimi_delta_attention",
            "key_head_dim": target["head_dim"],
            "value_head_dim": target["head_dim"],
            "num_key_heads": target["num_key_heads"],
            "num_value_heads": target["num_value_heads"],
            "conv_kernel_size": 4,
        }
    return {
        "block": {
            "hidden_size": hidden,
            "stages": [
                {"branches": [operation]},
                {
                    "branches": [
                        {"kind": "swiglu", "intermediate_size": target["intermediate_size"]}
                    ]
                },
            ],
        },
        "repeat": 3 if mixer == "kimi_delta_attention" else 1,
    }


def model_settings(target, vocab_size):
    """Expand one compact 3:1 target into an ArchitectureConfig-compatible dictionary."""

    cycles, remainder = divmod(target["depth"], 4)
    if remainder:
        raise ValueError("scale target depth must preserve the 3:1 mixer pattern")
    blocks = []
    for _ in range(cycles):
        blocks.extend((_block(target, "kimi_delta_attention"), _block(target, "attention")))
    return {
        "blocks": blocks,
        "embedding_size": target["embedding_size"],
        "vocab_size": vocab_size,
        "max_position_embeddings": 131_072,
        "rope_theta": 1_000_000.0,
        "rope_scaling_factor": 1.0,
        "rms_norm_eps": 1e-5,
        "initializer_range": 0.02,
    }


def parameter_accounting(target, vocab_size):
    """Count parameters from layer equations, independently of module construction."""

    embedding = target["embedding_size"]
    hidden = target["hidden_size"]
    intermediate = target["intermediate_size"]
    head = target["head_dim"]
    key_heads = target["num_key_heads"]
    value_heads = target["num_value_heads"]
    kv_heads = target["num_key_value_heads"]
    key_size = key_heads * head
    value_size = value_heads * head
    recurrent_layers = target["depth"] * 3 // 4
    global_layers = target["depth"] // 4

    recurrent_block = (
        hidden * (2 * key_size + 2 * value_size)
        + hidden * value_heads
        + hidden * head
        + head * value_heads * head
        + (2 * key_size + value_size) * 4
        + value_heads
        + value_heads * head
        + head
        + value_size * hidden
        + 2 * hidden
        + 3 * hidden * intermediate
    )
    global_block = (
        2 * hidden * hidden
        + 2 * hidden * kv_heads * head
        + 2 * head
        + 2 * hidden
        + 3 * hidden * intermediate
    )
    adapters = 0 if embedding == hidden else 2 * embedding * hidden
    components = {
        "shared_embedding_and_lm_head": vocab_size * embedding,
        "input_and_output_adapters": adapters,
        "recurrent_blocks": recurrent_layers * recurrent_block,
        "global_blocks": global_layers * global_block,
        "final_norm": hidden,
    }
    return {
        "formula": "V*E + adapters + R*P_kda + G*P_attn + H",
        "recurrent_block_formula": (
            "H(2K+2W)+H*Nv+H*D+D*Nv*D+(2K+W)*4+Nv+Nv*D+D+W*H+2H+3H*I"
        ),
        "global_block_formula": "2H^2+2H*Nkv*D+2D+2H+3H*I",
        "components": components,
        "recurrent_block_parameters": recurrent_block,
        "global_block_parameters": global_block,
        "total_parameters": sum(components.values()),
    }


def flop_accounting(target, vocab_size, length):
    """Reproduce the repository's projection-plus-sequence analytic training FLOPs."""

    embedding = target["embedding_size"]
    hidden = target["hidden_size"]
    intermediate = target["intermediate_size"]
    head = target["head_dim"]
    key_size = target["num_key_heads"] * head
    value_size = target["num_value_heads"] * head
    value_heads = target["num_value_heads"]
    kv_heads = target["num_key_value_heads"]
    recurrent_layers = target["depth"] * 3 // 4
    global_layers = target["depth"] // 4
    adapters = 0 if embedding == hidden else 2 * embedding * hidden

    recurrent_linear = (
        hidden * (2 * key_size + 3 * value_size)
        + hidden * (value_heads + head)
        + head * value_heads * head
        + (2 * key_size + value_size) * 4
        + 3 * hidden * intermediate
    )
    global_linear = (
        2 * hidden * hidden
        + 2 * hidden * kv_heads * head
        + 3 * hidden * intermediate
    )
    linear_macs = (
        vocab_size * embedding
        + adapters
        + recurrent_layers * recurrent_linear
        + global_layers * global_linear
    )
    chunk = min(64, length)
    recurrent = recurrent_layers * value_heads * (
        6 * head**2 + 3 * chunk * head + chunk**2
    )
    mean_causal_context = (length + 1) / 2
    global_attention = int(global_layers * 12 * mean_causal_context * hidden)
    total = 6 * linear_macs + recurrent + global_attention
    return {
        "length": length,
        "formula": "6*linear_projection_MACs + recurrent_rule_FLOPs + global_attention_FLOPs",
        "linear_projection_macs": linear_macs,
        "six_times_linear_projection_macs": 6 * linear_macs,
        "recurrent_rule_flops": recurrent,
        "global_attention_flops": global_attention,
        "analytic_training_flops_per_token": total,
    }


def optimizer_accounting(model):
    """Estimate initialized optimizer state from exact repository optimizer membership."""

    optimizer = model.optimizer(name="muon")
    roles = model.optimizer_role_counts(optimizer)
    adam_parameters = sum(
        value["parameters"] for name, value in roles.items() if name.startswith("adamw_")
    )
    adam_tensors = sum(
        value["tensors"] for name, value in roles.items() if name.startswith("adamw_")
    )
    muon_parameters = roles["muon"]["parameters"]
    return {
        "roles": roles,
        "assumption": (
            "all parameters have received gradients; no FP32 master weights; one Muon momentum "
            "buffer, two AdamW moment buffers, and one FP32 AdamW step scalar per tensor"
        ),
        "native_bf16_state_bytes": muon_parameters * 2 + adam_parameters * 4 + adam_tensors * 4,
        "fp32_state_counterfactual_bytes": (
            muon_parameters * 4 + adam_parameters * 8 + adam_tensors * 4
        ),
    }


def state_accounting(target, model):
    """Describe fixed KDA state and length-growing global KV state and validate both."""

    head = target["head_dim"]
    key_size = target["num_key_heads"] * head
    value_size = target["num_value_heads"] * head
    recurrent_layers = target["depth"] * 3 // 4
    global_layers = target["depth"] // 4
    recurrent_elements = recurrent_layers * target["num_value_heads"] * head * head
    convolution_elements = recurrent_layers * (2 * key_size + value_size) * 3
    points = []
    for length in LENGTHS:
        reports = {}
        for name, dtype in (("bf16", torch.bfloat16), ("int8_with_fp16_scales", torch.int8)):
            with torch.device("meta"):
                state = model.state(
                    length=length,
                    device="meta",
                    dtype=torch.bfloat16,
                    kv_cache_dtype=dtype,
                )
            reports[name] = state.memory_report()
        global_kv_elements = (
            global_layers * 2 * target["num_key_value_heads"] * length * head
        )
        expected = {
            "bf16": {
                "attention_kv": global_kv_elements * 2,
                "kimi_delta_attention": recurrent_elements * 4 + convolution_elements * 2,
            },
            "int8_with_fp16_scales": {
                "attention_kv": global_kv_elements
                + global_layers * 2 * target["num_key_value_heads"] * length * 2,
                "kimi_delta_attention": recurrent_elements * 4 + convolution_elements * 2,
            },
        }
        for name, by_kind in expected.items():
            if reports[name]["by_kind"] != by_kind or reports[name]["total_bytes"] != sum(
                by_kind.values()
            ):
                raise AssertionError(f"instantiated state mismatch for {target['id']} at {length}")
        points.append(
            {
                "length": length,
                "global_kv_elements": global_kv_elements,
                "instantiated_state": reports,
            }
        )
    return {
        "batch_size": 1,
        "recurrent_layers": recurrent_layers,
        "global_layers": global_layers,
        "recurrent_matrix_shape_per_layer": [1, target["num_value_heads"], head, head],
        "recurrent_matrix_dtype": "float32",
        "recurrent_matrix_elements": recurrent_elements,
        "recurrent_matrix_bytes": recurrent_elements * 4,
        "convolution_shape_per_layer": [1, 2 * key_size + value_size, 3],
        "convolution_dtype": "bfloat16",
        "convolution_elements": convolution_elements,
        "convolution_bytes": convolution_elements * 2,
        "fixed_recurrent_state_bytes": recurrent_elements * 4 + convolution_elements * 2,
        "global_kv_formula": "G*2*Nkv*L*D elements",
        "points": points,
    }


def tokenizer_costs(target, tokenizer):
    """Keep planned untied cost distinct from the currently shared instantiated storage."""

    width = target["embedding_size"]
    rows = [
        ("mistral-32k-fallback", tokenizer["base_vocab_size"]),
        ("speck-bpe-32000-whitespace", 32_000),
        ("speck-bpe-32768-whitespace", 32_768),
        ("speck-bpe-40960-whitespace", 40_960),
    ]
    return [
        {
            "id": identifier,
            "effective_vocab_size_with_chat_tokens": base_vocab + tokenizer["chat_added_tokens"],
            "planned_untied_embedding_and_head_parameters": (
                2 * (base_vocab + tokenizer["chat_added_tokens"]) * width
            ),
            "instantiated_shared_embedding_and_head_parameters": (
                (base_vocab + tokenizer["chat_added_tokens"]) * width
            ),
        }
        for identifier, base_vocab in rows
    ]


def generate_accounting(spec, root):
    """Generate the complete deterministic accounting artifact and instantiate every geometry."""

    tokenizer = spec["tokenizer_fallback"]
    plan_path = root / tokenizer["plan"]
    if hashlib.sha256(plan_path.read_bytes()).hexdigest() != tokenizer["plan_sha256"]:
        raise ValueError("tokenizer plan hash does not match the scale target contract")
    for path, digest in spec["supersedes"]["legacy_shape_a"].values():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"historical Shape-A identity changed: {path}")

    targets = []
    vocab_size = tokenizer["effective_vocab_size"]
    for target in spec["targets"]:
        settings = model_settings(target, vocab_size)
        analytic_parameters = parameter_accounting(target, vocab_size)
        settings["expected_parameters"] = analytic_parameters["total_parameters"]
        settings["expected_active_parameters"] = analytic_parameters["total_parameters"]
        with torch.device("meta"):
            model = build_model(settings, vocab_size=vocab_size)
        flops = [flop_accounting(target, vocab_size, length) for length in LENGTHS]
        for point in flops:
            if model.flops_per_token(point["length"]) != point["analytic_training_flops_per_token"]:
                raise AssertionError(f"analytic FLOP mismatch for {target['id']}")
        parameters = model.parameter_count()
        targets.append(
            {
                **target,
                "status": "planning_geometry_only_not_launchable",
                "effective_vocab_size": vocab_size,
                "recurrent_layers": target["depth"] * 3 // 4,
                "global_layers": target["depth"] // 4,
                "parameter_accounting": analytic_parameters,
                "active_parameters": model.active_parameter_count(),
                "six_nd_flops_per_token": 6 * parameters,
                "flop_accounting": flops,
                "optimizer_state_estimate": optimizer_accounting(model),
                "state_geometry": state_accounting(target, model),
                "tokenizer_embedding_and_head_costs": tokenizer_costs(target, tokenizer),
                "instantiation_validation": {
                    "device": "meta",
                    "parameter_count": parameters,
                    "active_parameter_count": model.active_parameter_count(),
                    "flops_match_at_lengths": list(LENGTHS),
                    "state_reports_match": True,
                },
            }
        )
    return {
        "format": FORMAT,
        "format_version": 1,
        "status": spec["status"],
        "source_spec": "research/flagship/targets/scale-targets-v1.json",
        "tokenizer_accounting": {
            **tokenizer,
            "authority": "fallback_accounting_only_D5_not_selected",
            "implementation_note": (
                "tokenizer v2 plans untied embedding/head cost, but current SpeckForCausalLM shares "
                "their storage; exact target totals follow the instantiated shared model"
            ),
        },
        "common_formulas": {
            "symbols": {
                "V": "effective vocabulary rows including three chat roles",
                "E": "embedding width",
                "H": "block hidden width",
                "I": "SwiGLU intermediate width",
                "D": "head dimension",
                "K": "num_key_heads * head_dim",
                "W": "num_value_heads * head_dim",
                "Nv": "KDA value heads",
                "Nkv": "global-attention KV heads",
                "R": "recurrent block count",
                "G": "global block count",
                "L": "sequence/cache length"
            },
            "six_nd_policy": (
                "6*N is reported only as a conventional comparator; it is not added to or "
                "substituted for the repository analytic FLOPs/token"
            ),
        },
        "targets": targets,
        "launch_authority": False,
        "requires_data_launch_authority": True,
        "missing_launch_inputs": spec["missing_launch_inputs"],
    }


def load_and_generate(spec_path, root=None):
    spec_path = Path(spec_path)
    root = Path(root) if root is not None else spec_path.parents[3]
    source = spec_path.read_bytes()
    result = generate_accounting(json.loads(source), root)
    result["source_spec_sha256"] = hashlib.sha256(source).hexdigest()
    return result
