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
    RoutedSwiGLUSpec,
    StageConfig,
    SwiGLUSpec,
)
from speck.model import (
    BatchedMuon,
    CausalLMTrainingOutput,
    CombinedOptimizer,
    DeviceAdamW,
    Linear,
    RoutedSwiGLU,
    SpeckForCausalLM,
    _grouped_expert_swiglu,
    _reference_expert_swiglu,
    build_model,
    causal_depthwise_conv1d,
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


def test_sliding_attention_cache_matches_full_forward():
    torch.manual_seed(2)
    model = model_with(AttentionSpec(4, 1, "sliding", 3), SwiGLUSpec(16))
    tokens = torch.randint(0, 16, (1, 8))
    state = model.state(length=8)
    assert state.allocated_bytes() == 1 * 1 * 3 * 4 * 4 * 2
    assert torch.allclose(model(tokens), cached_logits(model, tokens), atol=1e-5)


def test_convolution_state_matches_full_forward():
    torch.manual_seed(3)
    model = model_with(GatedCausalConvSpec(8, 3), SwiGLUSpec(16))
    tokens = torch.randint(0, 16, (1, 8))
    assert torch.allclose(model(tokens), cached_logits(model, tokens), atol=1e-5)


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


def test_muon_optimizer_routes_convolution_parameters_to_adamw():
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
    assert set(model.rotary) == {"4", "6"}


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


def test_routed_swiglu_forward_backward_and_training_output():
    torch.manual_seed(11)
    model = model_with(RoutedSwiGLUSpec(4, num_experts=4, top_k=2))
    tokens = torch.randint(0, 16, (2, 6))

    lm_loss = model(tokens, tokens)
    output = model(tokens, tokens, return_training_output=True)

    assert isinstance(output, CausalLMTrainingOutput)
    torch.testing.assert_close(output.lm_loss, lm_loss)
    torch.testing.assert_close(
        output.total_loss,
        output.lm_loss + 0.01 * output.load_balance_loss + 0.001 * output.z_loss,
    )
    assert len(output.routing) == 1
    stats = output.routing[0]
    assert stats.layer == "occurrence_0_stage_0_branch_0"
    assert stats.utilization.sum() == pytest.approx(1.0)
    assert stats.mean_probabilities.detach().sum().item() == pytest.approx(1.0)
    assert stats.utilization.mul(tokens.numel() * 2).round().sum() == tokens.numel() * 2
    output.total_loss.backward()
    operation = model.cores["group_0_repeat_0"].stages[0].branches[0].operation
    assert isinstance(operation, RoutedSwiGLU)
    assert all(
        parameter.grad is not None
        for parameter in (
            operation.router.weight,
            operation.gate_proj,
            operation.up_proj,
            operation.down_proj,
        )
    )


def test_routing_auxiliary_losses_match_their_definitions():
    operation = RoutedSwiGLU(2, RoutedSwiGLUSpec(2, num_experts=2, top_k=1))
    with torch.no_grad():
        operation.router.weight.copy_(torch.tensor([[3.0, 0.0], [-3.0, 0.0]]))
        for bank in (operation.gate_proj, operation.up_proj, operation.down_proj):
            bank.fill_(0.1)
    inputs = torch.tensor([[[1.0, 0.0]], [[-1.0, 0.0]]])

    _, stats = operation(inputs)
    logits = F.linear(inputs.reshape(-1, 2).float(), operation.router.weight.float())
    probabilities = logits.softmax(dim=-1)
    expected_utilization = torch.tensor([0.5, 0.5])

    torch.testing.assert_close(stats.mean_probabilities, probabilities.mean(dim=0))
    torch.testing.assert_close(stats.utilization, expected_utilization)
    torch.testing.assert_close(
        stats.load_balance_loss,
        2 * torch.sum(probabilities.mean(dim=0) * expected_utilization),
    )
    torch.testing.assert_close(stats.load_balance_loss, torch.tensor(1.0))
    torch.testing.assert_close(stats.z_loss, logits.logsumexp(dim=-1).square().mean())


def test_unbalanced_routing_has_large_load_balance_penalty():
    operation = RoutedSwiGLU(2, RoutedSwiGLUSpec(2, num_experts=4, top_k=1))
    with torch.no_grad():
        operation.router.weight.copy_(
            torch.tensor([[10.0, 10.0], [-10.0, -10.0], [-10.0, -10.0], [-10.0, -10.0]])
        )
    _, stats = operation(torch.ones(2, 3, 2))

    torch.testing.assert_close(stats.utilization, torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert stats.load_balance_loss.item() == pytest.approx(4.0)


def test_selected_top_k_logits_are_softmax_normalized_before_combining():
    operation = RoutedSwiGLU(2, RoutedSwiGLUSpec(2, num_experts=2, top_k=2))
    inputs = torch.tensor([[[-1.0, 0.5], [0.25, 2.0]]])
    with torch.no_grad():
        operation.router.weight.zero_()
        operation.gate_proj.copy_(
            torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[2.0, 0.0], [0.0, 2.0]],
                ]
            )
        )
        operation.up_proj.copy_(operation.gate_proj)
        operation.down_proj.copy_(operation.gate_proj)

    actual, _ = operation(inputs)
    experts = []
    flat = inputs.flatten(0, 1)
    for expert in range(2):
        hidden = F.silu(F.linear(flat, operation.gate_proj[expert]))
        hidden *= F.linear(flat, operation.up_proj[expert])
        experts.append(F.linear(hidden, operation.down_proj[expert]))
    expected = torch.stack(experts).mean(dim=0).view_as(inputs)

    torch.testing.assert_close(actual, expected)


def test_router_math_stays_fp32_with_bfloat16_expert_compute():
    operation = RoutedSwiGLU(8, RoutedSwiGLUSpec(8, num_experts=4, top_k=2))
    inputs = torch.randn(2, 3, 8, dtype=torch.bfloat16, requires_grad=True)

    output, stats = operation(inputs)
    output.float().square().mean().backward()

    assert output.dtype == torch.bfloat16
    assert stats.mean_probabilities.dtype == torch.float32
    assert stats.utilization.dtype == torch.float32
    assert stats.z_loss.dtype == torch.float32
    assert operation.router.weight.dtype == torch.float32
    assert operation.router.weight.grad.dtype == torch.float32


def test_routed_swiglu_incremental_logits_match_full_forward():
    torch.manual_seed(12)
    model = model_with(AttentionSpec(4, 1), RoutedSwiGLUSpec(4, 4, 2))
    tokens = torch.randint(0, 16, (1, 8))
    torch.testing.assert_close(model(tokens), cached_logits(model, tokens), atol=1e-5, rtol=1e-5)


def test_muon_updates_expert_bank_slices_independently_with_bank_state():
    torch.manual_seed(13)
    bank = torch.nn.Parameter(torch.randn(3, 8, 16))
    references = [torch.nn.Parameter(matrix.clone()) for matrix in bank.detach()]
    gradient = torch.randn_like(bank)
    bank.grad = gradient.clone()
    for parameter, matrix_gradient in zip(references, gradient):
        parameter.grad = matrix_gradient.clone()
    settings = {
        "lr": 1e-3,
        "weight_decay": 0.1,
        "adjust_lr_fn": "match_rms_adamw",
    }
    expected = torch.optim.Muon(references, **settings)
    actual = BatchedMuon([bank], **settings)

    expected.step()
    actual.step()

    torch.testing.assert_close(bank, torch.stack(references), rtol=0, atol=0)
    assert actual.state[bank]["momentum_buffer"].shape == bank.shape
    reloaded = BatchedMuon([torch.nn.Parameter(bank.detach().clone())], **settings)
    reloaded.load_state_dict(actual.state_dict())


def test_expert_banks_and_router_use_muon_while_convolution_stays_adamw():
    model = model_with(
        GatedCausalConvSpec(8, 3), RoutedSwiGLUSpec(4, num_experts=4, top_k=2)
    )
    optimizer = model.optimizer(name="muon")
    operation = model.cores["group_0_repeat_0"].stages[1].branches[0].operation
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

    assert isinstance(operation, RoutedSwiGLU)
    assert {
        id(operation.router.weight),
        id(operation.gate_proj),
        id(operation.up_proj),
        id(operation.down_proj),
    } <= muon_parameters
    convolution = model.cores["group_0_repeat_0"].stages[0].branches[0].operation
    assert id(convolution.kernel) in adamw_parameters
    assert muon_parameters.isdisjoint(adamw_parameters)
    assert muon_parameters | adamw_parameters == {id(parameter) for parameter in model.parameters()}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="grouped GEMM requires CUDA")
@pytest.mark.parametrize(
    ("num_experts", "intermediate_size"),
    ((8, 1152), (16, 576), (32, 576)),
)
def test_grouped_cuda_expert_output_and_gradients_match_reference(
    num_experts, intermediate_size
):
    torch.manual_seed(num_experts)
    hidden_size = 768
    counts = torch.randint(1, 17, (num_experts,), device="cuda", dtype=torch.int32)
    expert_ids = torch.repeat_interleave(
        torch.arange(num_experts, device="cuda"), counts.to(torch.int64)
    )
    inputs = torch.randn(
        int(counts.sum()), hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    reference_inputs = inputs.detach().clone().requires_grad_()
    banks = [
        torch.randn(
            num_experts,
            rows,
            columns,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        for rows, columns in (
            (intermediate_size, hidden_size),
            (intermediate_size, hidden_size),
            (hidden_size, intermediate_size),
        )
    ]
    reference_banks = [bank.detach().clone().requires_grad_() for bank in banks]

    actual = _grouped_expert_swiglu(inputs, counts, *banks)
    expected = _reference_expert_swiglu(reference_inputs, expert_ids, *reference_banks)
    actual.float().square().mean().backward()
    expected.float().square().mean().backward()

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=3e-2, atol=3e-2)
    for bank, reference in zip(banks, reference_banks):
        torch.testing.assert_close(bank.grad, reference.grad, rtol=3e-2, atol=3e-2)
