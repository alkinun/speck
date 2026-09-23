import importlib.metadata
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from speck.model import SpeckForCausalLM, build_model
from speck.model.architecture import (
    ArchitectureConfig,
    AttentionSpec,
    BlockConfig,
    BlockGroup,
    KimiDeltaAttentionSpec,
    StageConfig,
    SwiGLUSpec,
)
from speck.model.layers import (
    Linear,
    causal_depthwise_conv1d,
    torch_kimi_delta_rule,
)
from speck.training.optimizers import BatchedMuon, CombinedOptimizer, DeviceAdamW

REFERENCE_MODEL = Path(__file__).resolve().parents[2] / "experiments/qualification/model.json"
KDA = KimiDeltaAttentionSpec(4, 4, 1, 2, conv_kernel_size=3)


def _torch_gated_delta_rule(query, key, value, log_decay, beta):
    """Reference scalar-decay delta rule, the channel-constant special case of KDA."""

    query, key, value, log_decay, beta = (
        tensor.float() for tensor in (query, key, value, log_decay, beta)
    )
    query = query * torch.rsqrt(query.square().sum(dim=-1, keepdim=True) + 1e-6)
    key = key * torch.rsqrt(key.square().sum(dim=-1, keepdim=True) + 1e-6)
    query = query * (query.size(-1) ** -0.5)
    state = value.new_zeros(value.size(0), value.size(2), key.size(-1), value.size(-1))
    outputs = []
    for index in range(query.size(1)):
        state = state * log_decay[:, index].exp()[..., None, None]
        remembered = torch.einsum("bhkv,bhk->bhv", state, key[:, index])
        delta = (value[:, index] - remembered) * beta[:, index, :, None]
        state = state + torch.einsum("bhk,bhv->bhkv", key[:, index], delta)
        outputs.append(torch.einsum("bhkv,bhk->bhv", state, query[:, index]))
    return torch.stack(outputs, dim=1), state


def model_with(*stages, repeat=1, sharing="none"):
    block = BlockConfig(8, tuple(StageConfig((stage,)) for stage in stages))
    config = ArchitectureConfig(
        (BlockGroup(block, repeat=repeat, weight_sharing=sharing),),
        8,
        vocab_size=16,
        max_position_embeddings=16,
    )
    model = SpeckForCausalLM(config)
    model.init_weights()
    model.eval()
    return model


def cached_logits(model, tokens):
    state = model.state(length=tokens.size(1))
    values = [model(tokens[:, :1], state=state)]
    for index in range(1, tokens.size(1)):
        values.append(model(tokens[:, index : index + 1], state=state))
    return torch.cat(values, dim=1)


def chunked_logits(model, tokens, split):
    state = model.state(length=tokens.size(1))
    first = model(tokens[:, :split], state=state)
    second = model(tokens[:, split:], state=state)
    return torch.cat((first, second), dim=1)


def test_linear_applies_configured_bias():
    layer = Linear(2, 1, bias=True)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[2.0, 3.0]]))
        layer.bias.fill_(5.0)

    assert torch.equal(layer(torch.tensor([[7.0, 11.0]])), torch.tensor([[52.0]]))


def test_main_model_parameter_count():
    settings = json.loads(REFERENCE_MODEL.read_text())
    config = ArchitectureConfig.from_dict(settings)
    with torch.device("meta"):
        model = SpeckForCausalLM(config)
    assert model.parameter_count() == 1_195_884_576


def test_embedding_and_lm_head_are_one_physical_optimizer_parameter():
    model = model_with(SwiGLUSpec(16))
    embedding = model.embed_tokens.weight

    assert model.lm_head.weight is embedding
    assert (
        model.state_dict()["lm_head.weight"].data_ptr()
        == model.state_dict()["embed_tokens.weight"].data_ptr()
    )
    assert sum(parameter is embedding for parameter in model.parameters()) == 1

    optimizer = model.optimizer(name="muon")
    memberships = [
        (name, group["weight_decay"])
        for name, member in optimizer.optimizers.items()
        for group in member.param_groups
        for parameter in group["params"]
        if parameter is embedding
    ]
    assert memberships == [("adamw", 0.0)]


def test_tied_legacy_checkpoint_loads_strictly_and_contradictory_aliases_fail():
    source = model_with(SwiGLUSpec(16))
    state = source.state_dict()
    restored = model_with(SwiGLUSpec(16))

    loaded = restored.load_state_dict(state, strict=True)
    assert loaded.missing_keys == loaded.unexpected_keys == []
    assert restored.lm_head.weight is restored.embed_tokens.weight

    contradictory = {name: tensor.clone() for name, tensor in state.items()}
    contradictory["lm_head.weight"][0, 0] += 1
    with pytest.raises(RuntimeError, match="tensors are not tied"):
        restored.load_state_dict(contradictory, strict=True)


def test_model_rejects_unknown_loss_backend():
    config = model_with(SwiGLUSpec(16)).config
    with pytest.raises(ValueError, match="unsupported loss backend"):
        SpeckForCausalLM(config, loss_backend="unknown")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Liger requires CUDA")
@pytest.mark.parametrize("reduction", ("mean", "sum"))
def test_liger_loss_and_gradients_match_torch(reduction):
    pytest.importorskip("liger_kernel")
    torch.manual_seed(5)
    reference = model_with(SwiGLUSpec(16)).cuda()
    fused = SpeckForCausalLM(reference.config, loss_backend="liger").cuda()
    fused.load_state_dict(reference.state_dict())
    tokens = torch.randint(0, 16, (2, 8), device="cuda")
    targets = tokens.clone()
    targets[0, 0] = -100

    reference_loss = reference(tokens, targets, loss_reduction=reduction)
    fused_loss = torch.compile(fused, dynamic=False)(tokens, targets, loss_reduction=reduction)
    reference_loss.backward()
    fused_loss.backward()

    torch.testing.assert_close(fused_loss, reference_loss, rtol=2e-3, atol=2e-3)
    for reference_parameter, fused_parameter in zip(
        reference.parameters(), fused.parameters(), strict=False
    ):
        torch.testing.assert_close(
            fused_parameter.grad,
            reference_parameter.grad,
            rtol=2e-2,
            atol=2e-2,
        )


def test_global_attention_cache_matches_full_forward():
    torch.manual_seed(1)
    model = model_with(AttentionSpec(4, 1), SwiGLUSpec(16))
    tokens = torch.randint(0, 16, (1, 8))
    assert torch.allclose(model(tokens), cached_logits(model, tokens), atol=1e-5)
    assert torch.allclose(model(tokens), chunked_logits(model, tokens, 3), atol=1e-5)


@pytest.mark.parametrize("rope_dim", (0, 2))
def test_partial_and_nope_attention_cache_matches_full_forward(rope_dim):
    torch.manual_seed(11 + rope_dim)
    model = model_with(AttentionSpec(4, 1, rope_dim=rope_dim), SwiGLUSpec(16))
    tokens = torch.randint(0, 16, (1, 8))
    assert torch.allclose(model(tokens), chunked_logits(model, tokens, 5), atol=1e-5)


def test_rotary_memory_does_not_grow_with_context_length():
    config = model_with(AttentionSpec(4, 1)).config
    config = ArchitectureConfig(
        config.blocks,
        config.embedding_size,
        vocab_size=config.vocab_size,
        max_position_embeddings=1_000_000,
    )
    model = SpeckForCausalLM(config)
    buffers = tuple(model.rotary.buffers())
    assert sum(buffer.numel() for buffer in buffers) == 2
    assert torch.isfinite(model(torch.randint(0, 16, (1, 8)))).all()


def test_int8_kv_cache_tracks_scales_and_approximates_full_forward():
    torch.manual_seed(21)
    model = model_with(AttentionSpec(4, 1), SwiGLUSpec(16))
    tokens = torch.randint(0, 16, (1, 8))
    state = model.state(length=8, kv_cache_dtype=torch.int8)
    values = [model(tokens[:, :1], state=state)]
    for index in range(1, tokens.size(1)):
        values.append(model(tokens[:, index : index + 1], state=state))
    logits = torch.cat(values, dim=1)
    assert state.memory_report()["by_kind"]["attention_kv"] == 1 * 1 * 8 * (4 * 2 + 2 * 2)
    assert torch.allclose(model(tokens), logits, atol=2e-3, rtol=2e-3)


def test_channel_constant_kda_reduces_to_grouped_gated_deltanet():
    torch.manual_seed(43)
    query = torch.randn(2, 7, 1, 4)
    key = torch.randn_like(query)
    value = torch.randn(2, 7, 2, 4)
    scalar_decay = -torch.rand(2, 7, 2)
    channel_decay = scalar_decay[..., None].expand(2, 7, 2, 4)
    beta = torch.rand(2, 7, 2)

    expected = _torch_gated_delta_rule(
        query.repeat_interleave(2, dim=2),
        key.repeat_interleave(2, dim=2),
        value,
        scalar_decay,
        beta,
    )
    actual = torch_kimi_delta_rule(query, key, value, channel_decay, beta)

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


@pytest.mark.parametrize("activation", ("sigmoid", "silu"))
def test_kimi_delta_attention_state_matches_full_forward(activation):
    torch.manual_seed(47)
    spec = KimiDeltaAttentionSpec(4, 4, 1, 2, conv_kernel_size=3, output_gate_activation=activation)
    model = model_with(spec, SwiGLUSpec(16))
    tokens = torch.randint(0, 16, (1, 8))
    assert torch.allclose(model(tokens), cached_logits(model, tokens), atol=2e-5)


def test_kimi_delta_attention_state_is_independent_of_context_length():
    spec = KimiDeltaAttentionSpec(4, 4, 1, 2, conv_kernel_size=3)
    model = model_with(spec)
    short = model.state(length=4)
    long = model.state(length=16)
    expected = 1 * 2 * 4 * 4 * 4 + 1 * (2 * 4 + 2 * 4) * 2 * 4
    assert short.allocated_bytes() == expected
    assert long.allocated_bytes() == expected
    assert short.memory_report()["by_kind"] == {"kimi_delta_attention": expected}


@pytest.mark.parametrize("activation", ("sigmoid", "silu"))
def test_kimi_delta_attention_backward_is_finite(activation):
    spec = KimiDeltaAttentionSpec(4, 4, 1, 2, conv_kernel_size=3, output_gate_activation=activation)
    model = model_with(spec)
    tokens = torch.randint(0, 16, (2, 8))
    loss = model(tokens, tokens)
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_default_kimi_sigmoid_preserves_forward_backward_and_strict_checkpoint_loading():
    torch.manual_seed(49)
    legacy = model_with(KimiDeltaAttentionSpec(4, 4, 1, 2, conv_kernel_size=3))
    explicit = model_with(
        KimiDeltaAttentionSpec(4, 4, 1, 2, conv_kernel_size=3, output_gate_activation="sigmoid")
    )
    loaded = explicit.load_state_dict(legacy.state_dict(), strict=True)
    assert loaded.missing_keys == []
    assert loaded.unexpected_keys == []
    assert legacy.state_dict().keys() == explicit.state_dict().keys()

    tokens = torch.randint(0, 16, (2, 8))
    legacy_loss = legacy(tokens, tokens)
    explicit_loss = explicit(tokens, tokens)
    torch.testing.assert_close(explicit_loss, legacy_loss, rtol=0, atol=0)
    legacy_loss.backward()
    explicit_loss.backward()
    for legacy_parameter, explicit_parameter in zip(
        legacy.parameters(), explicit.parameters(), strict=False
    ):
        torch.testing.assert_close(explicit_parameter.grad, legacy_parameter.grad, rtol=0, atol=0)


def test_kimi_output_gate_activation_changes_only_behavior_not_geometry():
    torch.manual_seed(51)
    sigmoid = model_with(
        KimiDeltaAttentionSpec(4, 4, 1, 2, conv_kernel_size=3, output_gate_activation="sigmoid")
    )
    silu = model_with(
        KimiDeltaAttentionSpec(4, 4, 1, 2, conv_kernel_size=3, output_gate_activation="silu")
    )
    silu.load_state_dict(sigmoid.state_dict(), strict=True)
    tokens = torch.randint(0, 16, (2, 8))

    sigmoid_logits = sigmoid(tokens)
    silu_logits = silu(tokens)

    assert torch.isfinite(sigmoid_logits).all()
    assert torch.isfinite(silu_logits).all()
    assert not torch.allclose(sigmoid_logits, silu_logits)
    assert sigmoid.parameter_count() == silu.parameter_count()
    assert sigmoid.flops_per_token(16) == silu.flops_per_token(16)
    assert sigmoid.state(length=8).memory_report() == silu.state(length=8).memory_report()
    assert {name: tuple(tensor.shape) for name, tensor in sigmoid.state_dict().items()} == {
        name: tuple(tensor.shape) for name, tensor in silu.state_dict().items()
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FLA KDA requires CUDA")
@pytest.mark.parametrize("activation", ("sigmoid", "silu"))
def test_kimi_output_gates_cover_fla_chunk_backward_and_stateful_decode(activation):
    pytest.importorskip("fla.ops.kda")
    assert importlib.metadata.version("flash-linear-attention") == "0.5.0"
    torch.manual_seed(52)
    model = model_with(
        KimiDeltaAttentionSpec(4, 4, 1, 2, conv_kernel_size=3, output_gate_activation=activation)
    ).cuda()
    model.to(torch.bfloat16)
    tokens = torch.randint(0, 16, (1, 8), device="cuda")

    with torch.no_grad():
        full = model(tokens)
        decoded = cached_logits(model, tokens)
    torch.testing.assert_close(decoded, full, rtol=2e-2, atol=2e-2)

    model(tokens).float().square().mean().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_kimi_delta_attention_flops_follow_paper_chunk_formula():
    model = model_with(KimiDeltaAttentionSpec(4, 4, 1, 2))
    assert model.flops_per_token(16) == 4_256


def first_operation(model):
    return next(iter(model.cores.values())).stages[0].branches[0].operation


def test_kda_initialization_has_declared_time_ranges():
    torch.manual_seed(53)
    kda = first_operation(model_with(KimiDeltaAttentionSpec(4, 4, 1, 2)))

    kda_rates = kda.log_rates.exp()
    kda_dt = F.softplus(kda.decay_bias)

    assert (kda_rates >= 1).all() and (kda_rates <= 16).all()
    assert (kda_dt >= 0.001).all() and (kda_dt <= 0.1).all()


def test_activation_checkpointing_preserves_loss_and_gradients():
    torch.manual_seed(41)
    reference = model_with(AttentionSpec(4, 1), SwiGLUSpec(16))
    checkpointed = model_with(AttentionSpec(4, 1), SwiGLUSpec(16))
    checkpointed.load_state_dict(reference.state_dict())
    checkpointed.set_gradient_checkpointing(True)
    tokens = torch.randint(0, 16, (2, 8))
    reference_loss = reference(tokens, tokens)
    checkpointed_loss = checkpointed(tokens, tokens)
    reference_loss.backward()
    checkpointed_loss.backward()
    torch.testing.assert_close(checkpointed_loss, reference_loss)
    for expected, actual in zip(reference.parameters(), checkpointed.parameters(), strict=False):
        torch.testing.assert_close(actual.grad, expected.grad)


@pytest.mark.parametrize(
    ("kernel_size", "sequence_length"),
    ((3, 1), (5, 2), (3, 8), (5, 8)),
)
def test_direct_causal_convolution_matches_grouped_convolution(kernel_size, sequence_length):
    torch.manual_seed(kernel_size)
    inputs = torch.randn(2, 4, sequence_length, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()
    weight = torch.randn(4, 1, kernel_size, requires_grad=True)
    reference_weight = weight.detach().clone().requires_grad_()

    actual = causal_depthwise_conv1d(inputs, weight)
    expected = F.conv1d(
        F.pad(reference_inputs, (kernel_size - 1, 0)),
        reference_weight,
        groups=4,
    )
    actual.square().sum().backward()
    expected.square().sum().backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(inputs.grad, reference_inputs.grad)
    torch.testing.assert_close(weight.grad, reference_weight.grad)


def test_muon_optimizer_assigns_convolution_parameters_to_adamw():
    model = model_with(KDA, SwiGLUSpec(16))
    optimizer = model.optimizer(name="muon")
    assert isinstance(optimizer, CombinedOptimizer)
    assert isinstance(optimizer.optimizers["muon"], BatchedMuon)
    muon_parameters = {
        id(parameter)
        for group in optimizer.optimizers["muon"].param_groups
        for parameter in group["params"]
    }
    adamw_parameters = {
        id(parameter)
        for group in optimizer.optimizers["adamw"].param_groups
        for parameter in group["params"]
    }
    convolution_parameters = {
        id(parameter) for parameter in model.parameters() if parameter.ndim == 3
    }

    assert muon_parameters.isdisjoint(adamw_parameters)
    assert muon_parameters | adamw_parameters == {id(parameter) for parameter in model.parameters()}
    assert convolution_parameters <= adamw_parameters
    assert optimizer.optimizers["adamw"].param_groups[0]["weight_decay"] == 0.1
    roles = model.optimizer_role_counts(optimizer)
    assert sum(role["tensors"] for role in roles.values()) == len(tuple(model.parameters()))

    tokens = torch.randint(0, 16, (2, 8))
    model(tokens, tokens).backward()
    optimizer.step()
    state = optimizer.state_dict()
    optimizer.load_state_dict(state)


def test_batched_muon_matches_reference_and_keeps_compatible_state():
    torch.manual_seed(7)
    shapes = ((8, 16), (8, 16), (16, 8), (16, 8), (8, 8), (8, 8))
    reference_parameters = [torch.nn.Parameter(torch.randn(shape)) for shape in shapes]
    batched_parameters = [
        torch.nn.Parameter(parameter.detach().clone()) for parameter in reference_parameters
    ]
    gradients = [torch.randn_like(parameter) for parameter in reference_parameters]
    for reference, batched, gradient in zip(
        reference_parameters, batched_parameters, gradients, strict=False
    ):
        reference.grad = gradient.clone()
        batched.grad = gradient.clone()
    settings = {
        "lr": 1e-3,
        "weight_decay": 0.1,
        "adjust_lr_fn": "match_rms_adamw",
    }
    reference = torch.optim.Muon(reference_parameters, **settings)
    batched = BatchedMuon(batched_parameters, **settings)

    reference.step()
    batched.step()

    for expected, actual in zip(reference_parameters, batched_parameters, strict=False):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    reference.load_state_dict(batched.state_dict())
    batched.load_state_dict(reference.state_dict())


def test_batched_muon_step_compiles_with_a_tensor_learning_rate():
    parameter = torch.nn.Parameter(torch.randn(8, 8))
    parameter.grad = torch.randn_like(parameter)
    optimizer = BatchedMuon([parameter])
    optimizer.param_groups[0]["lr"] = torch.tensor(1e-3)
    graphs = []

    def backend(graph, _):
        graphs.append(graph)
        return graph.forward

    compiled_step = torch.compile(
        optimizer.step,
        backend=backend,
        fullgraph=True,
        dynamic=False,
    )

    compiled_step()

    assert len(graphs) == 1


def test_device_adamw_restores_runtime_fusion_after_loading_state():
    reference_parameter = torch.nn.Parameter(torch.ones(2))
    reference = torch.optim.AdamW([{"params": []}, {"params": [reference_parameter]}])
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = DeviceAdamW([{"params": []}, {"params": [parameter]}], fused=False)

    optimizer.load_state_dict(reference.state_dict())

    assert all(group["fused"] is False for group in optimizer.param_groups)


def test_shared_blocks_keep_occurrence_state_separate():
    torch.manual_seed(4)
    model = model_with(
        AttentionSpec(4, 1),
        KDA,
        repeat=2,
        sharing="all",
    )
    state = model.state(length=8)
    assert len(state.entries) == 4
    assert len(model.cores) == 1
    tokens = torch.randint(0, 16, (1, 8))
    assert torch.allclose(model(tokens), cached_logits(model, tokens), atol=1e-5)


def test_state_reset_replays_the_same_sequence():
    model = model_with(AttentionSpec(4, 1), KDA)
    state = model.state(length=8)
    tokens = torch.randint(0, 16, (1, 4))
    first = model(tokens, state=state)
    state.reset()
    second = model(tokens, state=state)
    assert torch.equal(first, second)


@pytest.mark.parametrize("length", (0, -1, 17, True, 1.5))
def test_state_rejects_invalid_lengths(length):
    model = model_with(AttentionSpec(4, 1))
    with pytest.raises(ValueError, match="state length"):
        model.state(length=length)


@pytest.mark.parametrize("batch_size", (0, -1, True, 1.5))
def test_state_rejects_invalid_batch_sizes(batch_size):
    model = model_with(AttentionSpec(4, 1))
    with pytest.raises(ValueError, match="batch size"):
        model.state(batch_size=batch_size)


def test_build_model_uses_block_config():
    model = model_with(SwiGLUSpec(16), repeat=2)
    rebuilt = build_model(model.config.export(), 16)
    assert isinstance(rebuilt, SpeckForCausalLM)
    assert rebuilt.parameter_count() == model.parameter_count()
    rebuilt.load_state_dict(model.state_dict())


def test_heterogeneous_head_dimensions_and_widths():
    config = ArchitectureConfig(
        (
            BlockGroup(BlockConfig(8, (StageConfig((AttentionSpec(4, 1),)),))),
            BlockGroup(BlockConfig(12, (StageConfig((AttentionSpec(6, 1),)),))),
        ),
        8,
        vocab_size=16,
        max_position_embeddings=8,
    )
    model = SpeckForCausalLM(config)
    model.init_weights()
    tokens = torch.randint(0, 16, (1, 6))
    assert torch.allclose(model(tokens), cached_logits(model, tokens), atol=1e-5)
    assert set(model.rotary) == {"4:4", "6:6"}


def test_parallel_stage_cache_matches_full_forward():
    config = ArchitectureConfig(
        (
            BlockGroup(
                BlockConfig(
                    8,
                    (
                        StageConfig(
                            (
                                AttentionSpec(4, 1),
                                KDA,
                            )
                        ),
                    ),
                )
            ),
        ),
        8,
        vocab_size=16,
        max_position_embeddings=8,
    )
    model = SpeckForCausalLM(config)
    model.init_weights()
    tokens = torch.randint(0, 16, (1, 6))
    assert torch.allclose(model(tokens), cached_logits(model, tokens), atol=1e-5)


@pytest.mark.parametrize("magnitude", (0.0, 1e-6, 1e-4, 1.0))
def test_int8_cache_quantizes_using_representable_nonzero_scales(magnitude):
    from speck.model import AttentionState

    state = AttentionState(1, 1, 2, 4, "cpu", torch.float32, storage_dtype=torch.int8)
    values = torch.tensor([[[[-1.0, -0.5, 0.5, 1.0]]]]) * magnitude
    state.append(values, values)
    keys, cached_values = state.current()
    assert torch.all(state.key_scales[:, :, :1] > 0)
    tolerance = max(magnitude / 127, 6e-8)
    torch.testing.assert_close(keys, values, rtol=0, atol=tolerance)
    torch.testing.assert_close(cached_values, values, rtol=0, atol=tolerance)


def test_resizing_embeddings_clears_both_parameter_count_expectations():
    from dataclasses import replace

    model = model_with(SwiGLUSpec(4))
    model.config = replace(
        model.config,
        expected_parameters=model.parameter_count(),
        expected_active_parameters=model.active_parameter_count(),
    )
    original = model.embed_tokens.weight.detach().clone()
    model.resize_token_embeddings(model.config.vocab_size + 3)
    assert model.config.expected_parameters is None
    assert model.config.expected_active_parameters is None
    assert model.lm_head.weight is model.embed_tokens.weight
    torch.testing.assert_close(model.embed_tokens.weight[: len(original)], original)
