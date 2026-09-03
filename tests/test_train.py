import copy
import math
from pathlib import Path

import pytest
import torch

from scripts.base_train import (
    BaseTrainer,
    arguments,
    changed_branch_settings,
    changed_context_settings,
    changed_resume_settings,
    context_compatible_architecture,
    validate,
)
from speck.architecture import (
    ArchitectureConfig,
    AttentionSpec,
    BlockConfig,
    BlockGroup,
    StageConfig,
    SwiGLUSpec,
)
from speck.config import load_experiment
from speck.model import SpeckForCausalLM
from speck.train import (
    assert_finite,
    branch_position,
    checkpoint_global_tokens,
    checkpoint_milestones,
    lr_scale,
    optimization_step,
    resolve_device_batch_size,
    validate_loader_progress,
)


def test_cpu_finite_check_rejects_non_finite_values():
    assert_finite(torch.tensor(1.0), "bad value")
    with pytest.raises(FloatingPointError, match="bad value"):
        assert_finite(torch.tensor(float("nan")), "bad value")


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


def test_lr_scale_rejects_unknown_schedule():
    with pytest.raises(ValueError):
        lr_scale(0, 10, 2, 0.1, "unknown")


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
        "global_token_offset": 0,
        "checkpoint_tokens": [],
        "training_phase": "base",
    }

    assert changed_resume_settings(legacy, current) == []
    assert changed_resume_settings(legacy, {**current, "lr_schedule": "constant"}) == [
        "lr_schedule"
    ]


def test_runtime_cadence_arguments_are_optional():
    defaults = arguments([])
    overridden = arguments(["experiment", "--save-every", "1526", "--eval-every", "0"])

    assert defaults.save_every is defaults.eval_every is None
    assert overridden.save_every == 1526
    assert overridden.eval_every == 0
    assert defaults.stop_at_tokens is None
    assert arguments(["experiment", "--stop-at-tokens", "50000000"]).stop_at_tokens == 50_000_000
    base = {
        "global_token_offset": 0,
        "checkpoint_tokens": [],
        "training_phase": "base",
    }
    assert not changed_resume_settings(
        {**base, "save_every": 10, "eval_every": 20},
        {**base, "save_every": 30, "eval_every": 40},
    )


def test_stop_at_tokens_is_restricted_to_configured_milestones(tmp_path):
    experiment = Path(__file__).parents[1] / "experiments" / "Speck1-140M"
    configs = load_experiment(experiment, "data", "tokenizer", "model", "train")
    configs["train"] = {
        **configs["train"],
        "output_dir": str(tmp_path),
        "checkpoint_tokens": [50_000_000, 500_000_000],
    }

    trainer = BaseTrainer(
        configs,
        arguments([str(experiment), "--stop-at-tokens", "50000000"]),
    )
    assert trainer.args.stop_at_tokens == 50_000_000
    assert trainer.args.seed == 42

    with pytest.raises(ValueError, match="configured checkpoint"):
        BaseTrainer(
            configs,
            arguments([str(experiment), "--stop-at-tokens", "100000000"]),
        )


def test_branch_schedule_argument_defaults_to_inherit():
    assert arguments([]).branch_schedule == "inherit"
    assert arguments(["--branch-schedule", "new"]).branch_schedule == "new"
    assert arguments([]).branch_kind == "same"
    assert arguments(["--branch-kind", "context"]).branch_kind == "context"


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


def test_context_branch_only_freezes_optimizer_semantics():
    previous = {
        "weight_decay": 0.1,
        "grad_clip": 1.0,
        "optimizer": "muon",
        "seed": 42,
        "sequence_length": 2_048,
        "world_size": 8,
    }
    current = {**previous, "sequence_length": 131_072, "world_size": 16}
    assert changed_context_settings(previous, current) == []
    assert changed_context_settings(previous, {**current, "optimizer": "adamw"}) == ["optimizer"]


def test_context_architecture_only_allows_positional_changes():
    experiment = Path(__file__).parents[1] / "experiments" / "Speck1-140M"
    model = load_experiment(experiment, "model")["model"]
    extended = {
        **model,
        "max_position_embeddings": 131_072,
        "rope_theta": 1_000_000.0,
        "rope_scaling_factor": 2.0,
    }
    assert context_compatible_architecture(model, extended)
    changed = {**extended, "embedding_size": model["embedding_size"] + 1}
    assert not context_compatible_architecture(model, changed)
    scope_changed = copy.deepcopy(extended)
    attention = scope_changed["blocks"][1]["block"]["stages"][0]["branches"][0]
    attention["scope"] = "sliding"
    attention["window_size"] = 32
    assert not context_compatible_architecture(model, scope_changed)
    assert context_compatible_architecture(
        model,
        scope_changed,
        allow_attention_scope_change=True,
    )
    rope_changed = copy.deepcopy(extended)
    rope_changed["blocks"][1]["block"]["stages"][0]["branches"][0]["rope_dim"] = 0
    assert not context_compatible_architecture(model, rope_changed)
    assert context_compatible_architecture(
        model,
        rope_changed,
        allow_attention_scope_change=True,
    )
    assert not context_compatible_architecture(
        model,
        changed,
        allow_attention_scope_change=True,
    )


def test_branch_inherits_global_and_schedule_positions():
    legacy = {
        "step": 30,
        "global_step": 30,
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
    assert checkpoint_global_tokens({"step": 30}, batch_tokens=10) == 300
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


def test_optimization_step_averages_accumulated_losses():
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
    optimizer = model.optimizer(name="muon")
    batches = [
        (torch.randint(0, 16, (1, 4)), torch.randint(0, 16, (1, 4)), {"batch": index})
        for index in range(3)
    ]
    with torch.no_grad():
        expected = torch.stack(
            [model(inputs, targets) for inputs, targets, _ in batches[:2]]
        ).mean()

    loss, grad_norm, next_batch = optimization_step(
        model,
        tuple(model.parameters()),
        optimizer,
        iter(batches[1:]),
        batches[0],
        accumulation=2,
        grad_clip=1.0,
        lr=1e-3,
    )

    torch.testing.assert_close(loss, expected)
    assert torch.isfinite(grad_norm)
    assert next_batch[2] == {"batch": 2}
