from types import SimpleNamespace

import pytest
import torch

from scripts.structured_retrieval_adapt import (
    build_supervised_batch,
    candidate_shift,
    replay_microsteps,
    validate_settings,
)
from tests.test_long_context import FakeTokenizer


def settings(**overrides):
    values = {
        "tasks": ("multi_key", "two_hop"),
        "sequence_length": 512,
        "steps": 10,
        "batch_size": 2,
        "accumulation": 2,
        "validation_samples": 4,
        "eval_every": 5,
        "records": 4,
        "chains": 4,
        "lr": 1e-4,
        "warmup_steps": 2,
        "min_lr": 0.1,
        "weight_decay": 0.1,
        "grad_clip": 1.0,
        "optimizer": "adamw",
        "seed": 42,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_supervised_batch_masks_every_position_except_answer():
    inputs, targets, cases = build_supervised_batch(
        FakeTokenizer(),
        ("multi_key", "two_hop"),
        sequence_length=1_024,
        batch_size=2,
        first_sample=1_000,
        records=4,
        chains=4,
        device=torch.device("cpu"),
    )
    assert inputs.shape == targets.shape == (2, 1_024)
    assert (targets != -100).sum().item() == 2
    assert targets[:, -1].tolist() == [case["answer_tokens"][0] for case in cases]
    assert [case["task"] for case in cases] == ["multi_key", "two_hop"]


def test_candidate_shift_is_symmetric():
    reference = torch.tensor([[2.0, 1.0], [0.0, 3.0]])
    changed = torch.tensor([[0.0, 4.0], [2.0, 1.0]])
    result = candidate_shift(
        reference,
        changed,
        torch.tensor([0, 1]),
        torch.tensor([1, 0]),
    )
    torch.testing.assert_close(result, torch.tensor([2.5, 2.0]))


def test_replay_microsteps_are_even_and_exact():
    assert replay_microsteps(4, 0.0) == ()
    assert replay_microsteps(4, 0.5) == (1, 3)
    assert replay_microsteps(4, 0.25) == (3,)
    with pytest.raises(ValueError, match="representable"):
        replay_microsteps(4, 0.3)


@pytest.mark.parametrize(
    "field,value",
    (("steps", 0), ("batch_size", True), ("lr", 0.0), ("min_lr", 2.0)),
)
def test_settings_reject_invalid_values(field, value):
    with pytest.raises(ValueError):
        validate_settings(settings(**{field: value}))
