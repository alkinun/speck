import pytest
import torch

from scripts.expert_masking import evaluate_masking, replay_loss
from speck.architecture import (
    ArchitectureConfig,
    BlockConfig,
    BlockGroup,
    RoutedSwiGLUSpec,
    StageConfig,
    SwiGLUSpec,
)
from speck.model import SpeckForCausalLM


def routed_model():
    config = ArchitectureConfig(
        (
            BlockGroup(
                BlockConfig(
                    8,
                    (
                        StageConfig((RoutedSwiGLUSpec(4, 4, 2),)),
                        StageConfig((RoutedSwiGLUSpec(4, 4, 2),)),
                    ),
                )
            ),
        ),
        embedding_size=8,
        vocab_size=16,
        max_position_embeddings=8,
    )
    model = SpeckForCausalLM(config)
    model.init_weights()
    model.eval()
    return model


def test_native_layer_mask_matches_zeroed_expert_banks():
    torch.manual_seed(21)
    model = routed_model()
    reference = routed_model()
    reference.load_state_dict(model.state_dict())
    layer = "occurrence_0_stage_0_branch_0"
    operation = reference.routed_operations()[layer]
    with torch.no_grad():
        operation.gate_proj.zero_()
        operation.up_proj.zero_()
        operation.down_proj.zero_()
    tokens = torch.randint(0, 16, (2, 6))

    masked = model(tokens, masked_routed_layers=(layer,))
    expected = reference(tokens)

    torch.testing.assert_close(masked, expected)
    with pytest.raises(ValueError, match="unknown routed layer"):
        model(tokens, masked_routed_layers=("missing",))


def test_masking_analysis_replays_identical_batches_and_reports_deltas():
    torch.manual_seed(22)
    model = routed_model()
    batches = tuple(
        (torch.randint(0, 16, (2, 4)), torch.randint(0, 16, (2, 4))) for _ in range(3)
    )

    baseline, results = evaluate_masking(model, batches, torch.device("cpu"))

    assert baseline == replay_loss(model, batches, torch.device("cpu"))
    assert [result["layer"] for result in results] == list(model.routed_operations())
    for result in results:
        expected = replay_loss(
            model,
            batches,
            torch.device("cpu"),
            masked_layer=result["layer"],
        )
        assert result["lm_loss"] == expected
        assert result["loss_delta"] == pytest.approx(expected - baseline)


def test_masking_analysis_rejects_dense_models():
    config = ArchitectureConfig(
        (
            BlockGroup(
                BlockConfig(
                    8,
                    (StageConfig((SwiGLUSpec(4),)),),
                )
            ),
        ),
        embedding_size=8,
        vocab_size=16,
        max_position_embeddings=8,
    )
    model = SpeckForCausalLM(config)
    model.init_weights()
    batches = ((torch.ones(1, 2, dtype=torch.long), torch.ones(1, 2, dtype=torch.long)),)

    with pytest.raises(ValueError, match="at least one routed"):
        evaluate_masking(model, batches, torch.device("cpu"))
