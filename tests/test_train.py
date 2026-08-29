import math

import pytest
import torch

from speck.architecture import (
    ArchitectureConfig,
    AttentionSpec,
    BlockConfig,
    BlockGroup,
    StageConfig,
    SwiGLUSpec,
)
from speck.model import SpeckForCausalLM
from speck.train import lr_scale, optimization_step, validate_loader_progress


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


@pytest.mark.parametrize(
    ("step", "steps", "warmup", "minimum"),
    (
        (-1, 10, 2, 0.1),
        (10, 10, 2, 0.1),
        (0, 0, 0, 0.1),
        (0, 10, 10, 0.1),
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
