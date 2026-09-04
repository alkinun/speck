import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from speck.architecture import (
    ArchitectureConfig,
    AttentionSpec,
    BlockConfig,
    BlockGroup,
    GatedCausalConvSpec,
    GatedDeltaNetSpec,
    KimiDeltaAttentionSpec,
    StageConfig,
    SwiGLUSpec,
)
from speck.model import (
    Attention,
    BatchedMuon,
    CombinedOptimizer,
    DeviceAdamW,
    Linear,
    SpeckForCausalLM,
    build_model,
    causal_attention_mask,
    causal_depthwise_conv1d,
    flex_sliding_window_attention,
    mean_causal_attention_context,
    sliding_window_block_mask,
    torch_gated_delta_rule,
    torch_kimi_delta_rule,
    torch_sliding_window_attention,
)

experiment = Path(__file__).parents[1] / "experiments" / "Speck1-140M"


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
    settings = json.loads((experiment / "model.json").read_text())
    config = ArchitectureConfig.from_dict(settings)
    model = SpeckForCausalLM(config)
    assert model.parameter_count() == 140_652_288


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
    for reference_parameter, fused_parameter in zip(reference.parameters(), fused.parameters()):
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


@pytest.mark.parametrize("output_gate", ("headwise", "elementwise"))
def test_gated_global_attention_cache_matches_full_forward(output_gate):
    torch.manual_seed(101)
    model = model_with(AttentionSpec(4, 1, rope_dim=0, output_gate=output_gate), SwiGLUSpec(16))
    tokens = torch.randint(0, 16, (1, 8))
    assert torch.allclose(model(tokens), cached_logits(model, tokens), atol=1e-5)
    assert torch.allclose(model(tokens), chunked_logits(model, tokens, 3), atol=1e-5)


@pytest.mark.parametrize("output_gate", ("headwise", "elementwise"))
def test_zero_initialized_attention_gate_halves_ungated_output(output_gate):
    torch.manual_seed(102)
    reference = Attention(8, AttentionSpec(4, 1, rope_dim=0), 1e-5)
    gated = Attention(8, AttentionSpec(4, 1, rope_dim=0, output_gate=output_gate), 1e-5)
    gated.load_state_dict(reference.state_dict(), strict=False)
    torch.nn.init.zeros_(gated.gate_proj.weight)
    values = torch.randn(2, 5, 8)
    torch.testing.assert_close(gated(values, None, 0), reference(values, None, 0) * 0.5)


def test_attention_gate_parameter_and_flop_accounting():
    ungated = model_with(AttentionSpec(4, 1, rope_dim=0))
    headwise = model_with(AttentionSpec(4, 1, rope_dim=0, output_gate="headwise"))
    elementwise = model_with(AttentionSpec(4, 1, rope_dim=0, output_gate="elementwise"))
    assert headwise.parameter_count() - ungated.parameter_count() == 8 * 2
    assert elementwise.parameter_count() - ungated.parameter_count() == 8 * 8
    assert headwise.flops_per_token(8) - ungated.flops_per_token(8) == 6 * 8 * 2
    assert elementwise.flops_per_token(8) - ungated.flops_per_token(8) == 6 * 8 * 8


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


def test_sliding_attention_cache_matches_full_forward():
    torch.manual_seed(2)
    model = model_with(AttentionSpec(4, 1, "sliding", 3), SwiGLUSpec(16))
    tokens = torch.randint(0, 16, (1, 8))
    state = model.state(length=8)
    assert state.allocated_bytes() == 1 * 1 * 3 * 4 * 4 * 2
    assert torch.allclose(model(tokens), cached_logits(model, tokens), atol=1e-5)


def test_sliding_attention_chunks_long_sequences():
    torch.manual_seed(22)
    model = model_with(AttentionSpec(4, 1, "sliding", 3), SwiGLUSpec(16))
    model.config = ArchitectureConfig(
        model.config.blocks,
        model.config.embedding_size,
        vocab_size=model.config.vocab_size,
        max_position_embeddings=2_100,
    )
    tokens = torch.randint(0, 16, (1, 2_100))
    output = model(tokens)
    assert output.shape == (1, 2_100, 16)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("with_state", (False, True))
def test_sliding_attention_skips_empty_kv_concatenation(monkeypatch, with_state):
    model = model_with(AttentionSpec(4, 1, "sliding", 3, rope_dim=0))
    tokens = torch.randint(0, 16, (1, 8))
    state = model.state(length=8) if with_state else None
    original_cat = torch.cat

    def reject_empty_prefix(tensors, *args, **kwargs):
        dim = kwargs.get("dim", args[0] if args else 0)
        if dim == 2 and len(tensors) == 2 and tensors[0].size(2) == 0:
            raise AssertionError("attention concatenated an empty K/V prefix")
        return original_cat(tensors, *args, **kwargs)

    monkeypatch.setattr(torch, "cat", reject_empty_prefix)
    model(tokens, state=state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention requires CUDA")
@pytest.mark.parametrize("past_length", (0, 3))
def test_flex_sliding_attention_matches_torch_reference(past_length):
    torch.manual_seed(23 + past_length)
    query = torch.randn(2, 4, 7, 16, device="cuda", requires_grad=True)
    key = torch.randn(
        2,
        2,
        7 + past_length,
        16,
        device="cuda",
        requires_grad=True,
    )
    value = torch.randn_like(key, requires_grad=True)

    compiled_flex = torch.compile(flex_sliding_window_attention, dynamic=False)
    actual = compiled_flex(query, key, value, past_length, window_size=4)
    expected = torch_sliding_window_attention(
        query,
        key,
        value,
        position=past_length,
        past_length=past_length,
        window_size=4,
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    actual_gradients = torch.autograd.grad(actual.float().square().sum(), (query, key, value))
    expected_gradients = torch.autograd.grad(expected.float().square().sum(), (query, key, value))
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-5, rtol=1e-5)


def test_long_sliding_block_mask_storage_scales_with_sparse_blocks():
    block_mask = sliding_window_block_mask(
        torch.device("cpu"),
        query_length=131_072,
        key_length=131_072,
        past_length=0,
        window_size=2_048,
    )
    tensors = (
        block_mask.kv_num_blocks,
        block_mask.kv_indices,
        block_mask.full_kv_num_blocks,
        block_mask.full_kv_indices,
        block_mask.q_num_blocks,
        block_mask.q_indices,
        block_mask.full_q_num_blocks,
        block_mask.full_q_indices,
    )
    stored_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    assert stored_bytes < 40 * 1_024**2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention requires CUDA")
def test_flex_sliding_attention_handles_non_block_aligned_model_prefill():
    config = ArchitectureConfig(
        (
            BlockGroup(
                BlockConfig(
                    64,
                    (StageConfig((AttentionSpec(16, 2, "sliding", 128),)),),
                )
            ),
        ),
        embedding_size=64,
        vocab_size=128,
        max_position_embeddings=256,
    )
    model = SpeckForCausalLM(config).cuda().eval()
    model.init_weights()
    state = model.state(length=256)

    output = model(torch.randint(0, 128, (1, 255), device="cuda"), state=state)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    ("sequence_length", "window_size"),
    ((1, None), (8, None), (8, 3), (3, 8)),
)
def test_attention_flops_context_matches_causal_mask(sequence_length, window_size):
    positions = torch.arange(sequence_length)
    mask = causal_attention_mask(positions[:, None], positions[None, :], window_size)

    expected = mask.sum().item() / sequence_length
    assert mean_causal_attention_context(sequence_length, window_size) == expected


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


def test_convolution_state_matches_full_forward():
    torch.manual_seed(3)
    model = model_with(GatedCausalConvSpec(8, 3), SwiGLUSpec(16))
    tokens = torch.randint(0, 16, (1, 8))
    assert torch.allclose(model(tokens), cached_logits(model, tokens), atol=1e-5)


def test_gated_deltanet_state_matches_full_forward():
    torch.manual_seed(31)
    spec = GatedDeltaNetSpec(4, 4, 1, 2, conv_kernel_size=3)
    model = model_with(spec, SwiGLUSpec(16))
    tokens = torch.randint(0, 16, (1, 8))
    assert torch.allclose(model(tokens), cached_logits(model, tokens), atol=2e-5)


def test_gated_deltanet_state_is_independent_of_context_length():
    spec = GatedDeltaNetSpec(4, 4, 1, 2, conv_kernel_size=3)
    model = model_with(spec)
    short = model.state(length=4)
    long = model.state(length=16)
    expected = 1 * 2 * 4 * 4 * 4 + 1 * (2 * 4 + 2 * 4) * 2 * 4
    assert short.allocated_bytes() == expected
    assert long.allocated_bytes() == expected


def test_gated_deltanet_backward_is_finite():
    spec = GatedDeltaNetSpec(4, 4, 1, 2, conv_kernel_size=3)
    model = model_with(spec)
    tokens = torch.randint(0, 16, (2, 8))
    loss = model(tokens, tokens)
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_gated_deltanet_output_gate_activation_is_configurable():
    torch.manual_seed(37)
    silu = model_with(GatedDeltaNetSpec(4, 4, 1, 2))
    sigmoid = model_with(GatedDeltaNetSpec(4, 4, 1, 2, output_gate_activation="sigmoid"))
    sigmoid.load_state_dict(silu.state_dict())
    tokens = torch.randint(0, 16, (2, 8))

    silu_logits = silu(tokens)
    sigmoid_logits = sigmoid(tokens)

    assert torch.isfinite(silu_logits).all()
    assert torch.isfinite(sigmoid_logits).all()
    assert not torch.allclose(sigmoid_logits, silu_logits)


def test_channel_constant_kda_reduces_to_grouped_gated_deltanet():
    torch.manual_seed(43)
    query = torch.randn(2, 7, 1, 4)
    key = torch.randn_like(query)
    value = torch.randn(2, 7, 2, 4)
    scalar_decay = -torch.rand(2, 7, 2)
    channel_decay = scalar_decay[..., None].expand(2, 7, 2, 4)
    beta = torch.rand(2, 7, 2)

    expected = torch_gated_delta_rule(
        query.repeat_interleave(2, dim=2),
        key.repeat_interleave(2, dim=2),
        value,
        scalar_decay,
        beta,
    )
    actual = torch_kimi_delta_rule(query, key, value, channel_decay, beta)

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_kimi_delta_attention_state_matches_full_forward():
    torch.manual_seed(47)
    spec = KimiDeltaAttentionSpec(4, 4, 1, 2, conv_kernel_size=3)
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


def test_kimi_delta_attention_backward_is_finite():
    spec = KimiDeltaAttentionSpec(4, 4, 1, 2, conv_kernel_size=3)
    model = model_with(spec)
    tokens = torch.randint(0, 16, (2, 8))
    loss = model(tokens, tokens)
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_kimi_delta_attention_flops_follow_paper_chunk_formula():
    model = model_with(KimiDeltaAttentionSpec(4, 4, 1, 2))
    assert model.flops_per_token(16) == 4_256


def first_operation(model):
    return next(iter(model.cores.values())).stages[0].branches[0].operation


def test_fla_delta_initialization_has_declared_time_ranges():
    torch.manual_seed(53)
    gdn = first_operation(model_with(GatedDeltaNetSpec(4, 4, 1, 2, decay_initialization="fla")))
    kda = first_operation(model_with(KimiDeltaAttentionSpec(4, 4, 1, 2)))

    gdn_rates = gdn.log_rates.exp()
    kda_rates = kda.log_rates.exp()
    gdn_dt = F.softplus(gdn.decay_bias)
    kda_dt = F.softplus(kda.decay_bias)

    assert (gdn_rates >= 1e-4).all() and (gdn_rates <= 16).all()
    assert (kda_rates >= 1).all() and (kda_rates <= 16).all()
    assert (gdn_dt >= 0.001).all() and (gdn_dt <= 0.1).all()
    assert (kda_dt >= 0.001).all() and (kda_dt <= 0.1).all()


def test_default_gdn_initialization_preserves_existing_speck_behavior():
    gdn = first_operation(model_with(GatedDeltaNetSpec(4, 4, 1, 2)))
    torch.testing.assert_close(gdn.log_rates.exp(), torch.tensor([0.1, 1.0]))
    assert torch.equal(gdn.decay_bias, torch.zeros(2))


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
    for expected, actual in zip(reference.parameters(), checkpointed.parameters()):
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
    model = model_with(GatedCausalConvSpec(8, 3), SwiGLUSpec(16))
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
    for reference, batched, gradient in zip(reference_parameters, batched_parameters, gradients):
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

    for expected, actual in zip(reference_parameters, batched_parameters):
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
        AttentionSpec(4, 1, "sliding", 3),
        GatedCausalConvSpec(8, 3),
        repeat=2,
        sharing="all",
    )
    state = model.state(length=8)
    assert len(state.entries) == 4
    assert len(model.cores) == 1
    tokens = torch.randint(0, 16, (1, 8))
    assert torch.allclose(model(tokens), cached_logits(model, tokens), atol=1e-5)


def test_state_reset_replays_the_same_sequence():
    model = model_with(AttentionSpec(4, 1), GatedCausalConvSpec(8, 3))
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
    assert set(model.rotary) == {"global:4:4", "global:6:6"}


def test_mixed_attention_scopes_use_independent_rope_scaling():
    config = ArchitectureConfig(
        (
            BlockGroup(BlockConfig(8, (StageConfig((AttentionSpec(4, 1),)),))),
            BlockGroup(BlockConfig(8, (StageConfig((AttentionSpec(4, 1, "sliding", 3),)),))),
        ),
        8,
        vocab_size=16,
        max_position_embeddings=8,
        rope_scaling_factor=4.0,
    )
    model = SpeckForCausalLM(config)

    assert model.rotary["global:4:4"].scaling_factor == 4.0
    assert model.rotary["sliding:4:4"].scaling_factor == 1.0


def test_parallel_stage_cache_matches_full_forward():
    config = ArchitectureConfig(
        (
            BlockGroup(
                BlockConfig(
                    8,
                    (
                        StageConfig(
                            (
                                AttentionSpec(4, 1, "sliding", 3),
                                GatedCausalConvSpec(8, 3),
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


def memory_model(*layers, length=16):
    groups = tuple(
        BlockGroup(BlockConfig(8, (StageConfig((layer,)), StageConfig((SwiGLUSpec(16),)))))
        for layer in layers
    )
    config = ArchitectureConfig(groups, 8, vocab_size=16, max_position_embeddings=length)
    model = SpeckForCausalLM(config)
    model.init_weights()
    model.eval()
    return model


def memory_specs(rope_dim=0, num_key_value_heads=1):
    writer = AttentionSpec(
        4, num_key_value_heads, rope_dim=rope_dim, memory="global", memory_role="write"
    )
    reader = AttentionSpec(
        4, num_key_value_heads, rope_dim=rope_dim, memory="global", memory_role="read"
    )
    standard = AttentionSpec(4, num_key_value_heads, rope_dim=rope_dim)
    return writer, reader, standard


def block_modules(model):
    return [model.cores[invocation.weight_key] for invocation in model.execution_plan]


@pytest.mark.parametrize("rope_dim", (0, 4))
def test_reader_attention_reads_exactly_the_written_keys_and_values(rope_dim):
    torch.manual_seed(210 + rope_dim)
    writer, reader, standard = memory_specs(rope_dim=rope_dim)
    shared = memory_model(writer, reader)
    reference = memory_model(standard, standard)
    reference.load_state_dict(shared.state_dict(), strict=False)
    written = block_modules(shared)[0].stages[0].branches[0].operation
    borrowed = block_modules(reference)[1].stages[0].branches[0].operation
    borrowed.k_proj.load_state_dict(written.k_proj.state_dict())
    borrowed.v_proj.load_state_dict(written.v_proj.state_dict())
    borrowed.k_norm.load_state_dict(written.k_norm.state_dict())
    for blocks in (block_modules(shared), block_modules(reference)):
        torch.nn.init.zeros_(blocks[0].stages[0].branches[0].operation.o_proj.weight)
        torch.nn.init.zeros_(blocks[0].stages[1].branches[0].operation.down_proj.weight)
    tokens = torch.randint(0, 16, (2, 6))
    torch.testing.assert_close(shared(tokens), reference(tokens))


def test_reader_attention_requires_a_written_memory():
    layer = Attention(8, AttentionSpec(4, 1, rope_dim=0, memory="global", memory_role="read"), 1e-5)
    with pytest.raises(ValueError, match="has not been written"):
        layer(torch.randn(1, 3, 8), None, 0)


@pytest.mark.parametrize("rope_dim", (0, 4))
def test_reader_attention_cache_matches_full_forward(rope_dim):
    torch.manual_seed(220 + rope_dim)
    writer, reader, _ = memory_specs(rope_dim=rope_dim)
    model = memory_model(writer, reader, reader)
    tokens = torch.randint(0, 16, (1, 8))
    assert torch.allclose(model(tokens), cached_logits(model, tokens), atol=1e-5)
    assert torch.allclose(model(tokens), chunked_logits(model, tokens, 3), atol=1e-5)


def test_reader_attention_allocates_one_shared_key_value_cache():
    writer, reader, standard = memory_specs()
    shared = memory_model(writer, reader, reader, reader)
    independent = memory_model(standard, standard, standard, standard)
    single = memory_model(standard)
    assert len(shared.state(length=16).entries) == 1
    assert len(independent.state(length=16).entries) == 4
    single_bytes = single.state(length=16).allocated_bytes()
    assert shared.state(length=16).allocated_bytes() == single_bytes
    assert independent.state(length=16).allocated_bytes() == 4 * single_bytes


def test_reader_attention_drops_key_value_projections_from_parameters_and_flops():
    writer, reader, standard = memory_specs()
    shared = memory_model(writer, reader, reader, reader)
    independent = memory_model(standard, standard, standard, standard)
    key_value_parameters = 2 * 8 * (1 * 4)
    key_norm_parameters = 4
    assert independent.parameter_count() - shared.parameter_count() == 3 * (
        key_value_parameters + key_norm_parameters
    )
    assert independent.flops_per_token(16) - shared.flops_per_token(16) == (
        6 * 3 * key_value_parameters
    )


def test_reader_attention_keeps_the_full_attention_score_term():
    writer, reader, standard = memory_specs()
    shared = memory_model(writer, reader)
    independent = memory_model(standard, standard)
    assert shared.flops_per_token(16) - shared.flops_per_token(8) == (
        independent.flops_per_token(16) - independent.flops_per_token(8)
    )
    assert shared.flops_per_token(16) > shared.flops_per_token(8)


def test_activation_checkpointing_preserves_reader_loss_and_gradients():
    torch.manual_seed(230)
    writer, reader, _ = memory_specs()
    reference = memory_model(writer, reader, reader)
    checkpointed = memory_model(writer, reader, reader)
    checkpointed.load_state_dict(reference.state_dict())
    reference.train()
    checkpointed.train()
    checkpointed.set_gradient_checkpointing(True)
    tokens = torch.randint(0, 16, (2, 8))
    reference_loss = reference(tokens, tokens)
    checkpointed_loss = checkpointed(tokens, tokens)
    reference_loss.backward()
    checkpointed_loss.backward()
    torch.testing.assert_close(checkpointed_loss, reference_loss)
    gradients = 0
    for expected, actual in zip(reference.parameters(), checkpointed.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad)
        gradients += int(actual.grad is not None and actual.grad.abs().sum() > 0)
    assert gradients


def test_reader_gradients_reach_the_memory_writer():
    torch.manual_seed(240)
    writer, reader, _ = memory_specs()
    model = memory_model(writer, reader)
    model.train()
    written = block_modules(model)[0].stages[0].branches[0].operation
    torch.nn.init.zeros_(written.o_proj.weight)
    torch.nn.init.zeros_(block_modules(model)[0].stages[1].branches[0].operation.down_proj.weight)
    tokens = torch.randint(0, 16, (2, 6))
    model(tokens, tokens).backward()
    assert written.k_proj.weight.grad.abs().sum() > 0
    assert written.v_proj.weight.grad.abs().sum() > 0


def test_shared_memory_key_adapters_are_absorbed_by_the_query_projection():
    torch.manual_seed(250)
    query = torch.randn(2, 3, 5, 4)
    key = torch.randn(2, 3, 7, 4)
    adapter = torch.randn(4, 4) * 0.1
    rekeyed = key + key @ adapter.transpose(0, 1)
    requeried = query + query @ adapter
    torch.testing.assert_close(
        query @ rekeyed.transpose(-1, -2),
        requeried @ key.transpose(-1, -2),
    )
