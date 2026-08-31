import math

import pytest
import torch

from scripts.base_train import changed_resume_settings
from speck.architecture import (
    ArchitectureConfig,
    AttentionSpec,
    BlockConfig,
    BlockGroup,
    StageConfig,
    SwiGLUSpec,
)
from speck.model import SpeckForCausalLM
from speck.train import (
    checkpoint_milestones,
    lr_scale,
    optimization_step,
    resolve_device_batch_size,
    validate_loader_progress,
)


def test_lr_scale_reaches_minimum_on_last_executed_step():
    assert lr_scale(0, 10, 2, 0.1) == 0.5
    assert lr_scale(1, 10, 2, 0.1) == 1.0
    assert lr_scale(2, 10, 2, 0.1) < 1.0
    assert lr_scale(9, 10, 2, 0.1) == 0.1
    assert all(
        left >= right
        for left, right in zip(
            (lr_scale(step, 10, 2, 0.1) for step in range(1, 9)),
            (lr_scale(step, 10, 2, 0.1) for step in range(2, 10)),
        )
    )


def test_lr_scale_without_warmup_starts_at_peak_and_ends_at_minimum():
    assert lr_scale(0, 4, 0, 0.1) == 1.0
    assert lr_scale(3, 4, 0, 0.1) == 0.1


def test_single_step_lr_schedule_uses_minimum():
    assert lr_scale(0, 1, 0, 0.25) == 0.25


def test_short_phase_can_end_during_warmup():
    assert lr_scale(0, 500, 512, 0.1) == 1 / 512
    assert lr_scale(499, 500, 512, 0.1) == 500 / 512


@pytest.mark.parametrize(
    ("world_size", "expected"),
    ((1, 16), (2, 16), (4, 8), (8, 4)),
)
def test_device_batch_ceiling_resolves_for_distributed_training(world_size, expected):
    assert resolve_device_batch_size(16, 65_536, 2_048, world_size) == expected


@pytest.mark.parametrize(
    ("limit", "batch_tokens", "sequence_length", "world_size"),
    ((0, 65_536, 2_048, 1), (16, 65_536, 2_048, 3), (12, 65_536, 2_048, 1)),
)
def test_device_batch_resolution_rejects_invalid_geometry(
    limit, batch_tokens, sequence_length, world_size
):
    with pytest.raises(ValueError, match="batch"):
        resolve_device_batch_size(limit, batch_tokens, sequence_length, world_size)


@pytest.mark.parametrize(
    ("step", "steps", "warmup", "minimum"),
    (
        (-1, 10, 2, 0.1),
        (10, 10, 2, 0.1),
        (0, 0, 0, 0.1),
        (0, 10, -1, 0.1),
        (0, 10, 2, math.nan),
        (0, 10, 2, -0.1),
        (0, 10, 2, 1.1),
    ),
)
def test_lr_scale_rejects_invalid_schedules(step, steps, warmup, minimum):
    with pytest.raises(ValueError):
        lr_scale(step, steps, warmup, minimum)


def test_checkpoint_loader_progress_matches_next_batch_offset():
    validate_loader_progress({"global_consumed_tokens": 65_536}, 65_536)
    try:
        validate_loader_progress({"global_consumed_tokens": 32_768}, 65_536)
    except ValueError as error:
        assert "does not match training progress" in str(error)
    else:
        raise AssertionError("mismatched loader progress was accepted")


def test_legacy_resume_defaults_to_torch_loss_backend():
    legacy = {}
    current = {
        "loss_backend": "torch",
        "global_token_offset": 0,
        "checkpoint_tokens": [],
        "training_phase": "base",
    }

    assert changed_resume_settings(legacy, current) == []
    assert changed_resume_settings(legacy, {**current, "loss_backend": "liger"}) == ["loss_backend"]


def test_checkpoint_milestones_align_baseline_and_warmup_runs():
    requested = [50_000_000, 100_000_000, 250_000_000, 500_000_000]
    baseline = checkpoint_milestones(requested, 65_536, 0, 7_630)
    initialized = checkpoint_milestones(requested, 65_536, 32_768_000, 7_130)

    assert baseline == {763: 50_000_000, 1_526: 100_000_000, 3_815: 250_000_000, 7_630: 500_000_000}
    assert initialized == {
        263: 50_000_000,
        1_026: 100_000_000,
        3_315: 250_000_000,
        7_130: 500_000_000,
    }
    for baseline_step, initialized_step in zip(baseline, initialized):
        assert baseline_step * 65_536 == 32_768_000 + initialized_step * 65_536


def test_optimization_step_advances_the_loader():
    config = ArchitectureConfig(
        (
            BlockGroup(
                BlockConfig(
                    8,
                    (
                        StageConfig((AttentionSpec(4, 1),)),
                        StageConfig((SwiGLUSpec(16),)),
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
    optimizer = model.optimizer()
    first = (torch.randint(0, 16, (1, 4)), torch.randint(0, 16, (1, 4)), {"batch": 0})
    second = (torch.randint(0, 16, (1, 4)), torch.randint(0, 16, (1, 4)), {"batch": 1})
    loader = iter([second])

    loss, grad_norm, next_batch = optimization_step(
        model,
        tuple(model.parameters()),
        optimizer,
        loader,
        first,
        accumulation=1,
        grad_clip=1.0,
        lr=1e-3,
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(grad_norm)
    assert next_batch[2] == {"batch": 1}
    assert optimizer.param_groups[0]["lr"] == 1e-3
