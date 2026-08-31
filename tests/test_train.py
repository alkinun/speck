import math

import pytest
import torch

from scripts.base_train import arguments, changed_branch_settings, changed_resume_settings, validate
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
    UpdateMonitor,
    branch_position,
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


def test_constant_lr_schedule_stays_at_peak():
    assert [lr_scale(step, 3, 0, 1.0, "constant") for step in range(3)] == [1.0] * 3
    with pytest.raises(ValueError, match="constant"):
        lr_scale(0, 3, 1, 1.0, "constant")
    with pytest.raises(ValueError, match="constant"):
        lr_scale(0, 3, 0, 0.1, "constant")


def test_single_step_lr_schedule_uses_minimum():
    assert lr_scale(0, 1, 0, 0.25) == 0.25


def test_short_phase_can_end_during_warmup():
    assert lr_scale(0, 500, 512, 0.1) == 1 / 512
    assert lr_scale(499, 500, 512, 0.1) == 500 / 512


def test_wsd_warms_up_stays_stable_and_decays_linearly():
    scales = [lr_scale(step, 10, 2, 0.1, "wsd", 3) for step in range(10)]

    assert scales[:2] == [0.5, 1.0]
    assert scales[2:8] == [1.0] * 6
    assert scales[8] == pytest.approx(0.55)
    assert scales[9] == 0.1
    assert lr_scale(3, 4, 0, 0.2, "wsd", 1) == 0.2


@pytest.mark.parametrize(
    ("schedule", "decay_steps"),
    (("unknown", None), ("cosine", 2), ("wsd", None), ("wsd", 0), ("wsd", 9)),
)
def test_lr_scale_rejects_invalid_schedule_settings(schedule, decay_steps):
    with pytest.raises(ValueError):
        lr_scale(0, 10, 2, 0.1, schedule, decay_steps)


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


def test_legacy_resume_defaults_to_cosine_schedule():
    legacy = {}
    current = {
        "lr_schedule": "cosine",
        "decay_steps": None,
        "global_token_offset": 0,
        "checkpoint_tokens": [],
        "training_phase": "base",
    }

    assert changed_resume_settings(legacy, current) == []
    assert changed_resume_settings(legacy, {**current, "lr_schedule": "wsd"}) == ["lr_schedule"]


def test_runtime_cadence_arguments_are_optional():
    defaults = arguments([])
    overridden = arguments(["experiment", "--save-every", "1526", "--eval-every", "0"])

    assert defaults.save_every is defaults.eval_every is None
    assert overridden.save_every == 1526
    assert overridden.eval_every == 0
    base = {
        "global_token_offset": 0,
        "checkpoint_tokens": [],
        "training_phase": "base",
    }
    assert not changed_resume_settings(
        {**base, "save_every": 10, "eval_every": 20},
        {**base, "save_every": 30, "eval_every": 40},
    )


def test_branch_schedule_argument_defaults_to_inherit():
    assert arguments([]).branch_schedule == "inherit"
    assert arguments(["--branch-schedule", "new"]).branch_schedule == "new"


def test_branch_only_allows_same_training_recipe():
    previous = {"lr": 1e-3, "world_size": 1}
    assert changed_branch_settings(previous, dict(previous)) == []
    assert changed_branch_settings(previous, {**previous, "lr": 2e-3}) == ["lr"]
    assert not changed_branch_settings(
        previous, {**previous, "lr": 2e-3}, allow_schedule_change=True
    )
    assert changed_branch_settings(
        previous, {**previous, "world_size": 2}, allow_schedule_change=True
    ) == ["world_size"]


def test_branch_inherits_global_and_schedule_positions():
    legacy = {
        "step": 30,
        "global_step": 30,
        "global_tokens": 300,
        "data_state": {"global_consumed_tokens": 300},
        "resolved": {"steps": 200},
    }
    metadata = {
        "step": 30,
        "global_step": 130,
        "global_tokens": 1300,
        "data_state": {"global_consumed_tokens": 300},
        "resolved": {"steps": 200, "schedule_step_offset": 100, "schedule_steps": 300},
    }

    assert branch_position(legacy, batch_tokens=10, steps=40) == (300, 300, 30, 200)
    assert branch_position(metadata, batch_tokens=10, steps=40) == (1300, 300, 130, 300)
    assert branch_position(metadata, batch_tokens=10) == (1300, 300, 130, 300)
    with pytest.raises(ValueError, match="exceeds"):
        branch_position(metadata, batch_tokens=10, steps=171)


def test_validation_reports_aggregate_and_source_losses():
    class MeanInput(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(self, inputs, targets):
            return inputs.float().mean() + 0 * self.anchor

    model = MeanInput()
    loader = iter(
        [
            (torch.tensor([[1.0]]), None, {"selected_source": "web"}),
            (torch.tensor([[4.0]]), None, {"selected_source": "math"}),
            (torch.tensor([[3.0]]), None, {"selected_source": "web"}),
        ]
    )

    loss, source_losses = validate(model, loader, 3, 1, ("web", "math"))

    assert loss == pytest.approx(8 / 3)
    assert source_losses == pytest.approx({"web": 2.0, "math": 4.0})
    assert model.training


def test_update_monitor_measures_relative_parameter_change():
    first = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    second = torch.nn.Parameter(torch.tensor([0.0, 2.0]))
    monitor = UpdateMonitor((("first", first), ("second", second)))
    snapshot = monitor.snapshot()

    with torch.no_grad():
        first.add_(torch.tensor([0.0, 5.0]))
        second.add_(torch.tensor([0.0, 1.0]))
    metrics = monitor.metrics(snapshot)

    assert monitor.names == ("first", "second")
    assert metrics["first"] == pytest.approx(
        {"weight_norm": 5.0, "update_norm": 5.0, "effective_lr": 1.0}
    )
    assert metrics["second"] == pytest.approx(
        {"weight_norm": 2.0, "update_norm": 1.0, "effective_lr": 0.5}
    )


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
