from types import SimpleNamespace

import pytest
import torch

from scripts.structured_retrieval_adapt import (
    build_supervised_batch,
    candidate_ranking_loss,
    candidate_shift,
    parse_adaptation_tasks,
    parse_answer_sets,
    parse_record_counts,
    parse_templates,
    replay_microsteps,
    training_tasks_for_step,
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


def test_supervised_batch_trains_every_multi_token_answer_position():
    class WordTokenizer:
        bos_id = 1

        def __init__(self):
            self.ids = {}

        def encode(self, text, bos=False):
            tokens = []
            for word in text.replace("\n", " \n ").split():
                if word not in self.ids:
                    self.ids[word] = len(self.ids) + 3
                tokens.append(self.ids[word])
            return ([self.bos_id] if bos else []) + tokens

    inputs, targets, cases = build_supervised_batch(
        WordTokenizer(),
        ("multi_key",),
        sequence_length=512,
        batch_size=2,
        first_sample=1_000,
        records=4,
        chains=4,
        device=torch.device("cpu"),
        templates=("archive", "registry"),
        answer_sets=("phrases",),
    )
    assert inputs.shape == targets.shape == (2, 512)
    assert (targets != -100).sum().item() == 4
    assert [case["template"] for case in cases] == ["archive", "registry"]
    for row, case in enumerate(cases):
        start = case["prompt_length"] - 1
        assert targets[row, start : start + 2].tolist() == case["answer_tokens"]


def test_template_and_answer_set_lists_are_strict():
    assert parse_templates("archive,registry") == ("archive", "registry")
    assert parse_answer_sets("letters,phrases") == ("letters", "phrases")
    with pytest.raises(ValueError, match="templates"):
        parse_templates("archive,archive")
    with pytest.raises(ValueError, match="answer sets"):
        parse_answer_sets("unknown")
    assert parse_record_counts("2,8") == (2, 8)
    with pytest.raises(ValueError, match="record counts"):
        parse_record_counts("8,8")
    assert parse_adaptation_tasks(
        "two_hop_route,two_hop_payload,two_hop_symbolic,two_hop_chain"
    ) == (
        "two_hop_route",
        "two_hop_payload",
        "two_hop_symbolic",
        "two_hop_chain",
    )


def test_supervised_batch_cycles_over_record_loads():
    _, _, cases = build_supervised_batch(
        FakeTokenizer(),
        ("multi_key",),
        sequence_length=1_024,
        batch_size=2,
        first_sample=0,
        records=8,
        chains=4,
        device=torch.device("cpu"),
        record_counts=(2, 8),
        response_cue="answer",
    )
    assert [case["records"] for case in cases] == [2, 8]
    assert [case["response_cue"] for case in cases] == ["answer", "answer"]


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


def test_candidate_ranking_loss_uses_each_response_position_and_vocabulary():
    hidden = torch.zeros(2, 4, 3)
    hidden[0, 2, 0] = 1
    hidden[1, 1, 1] = 1
    weight = torch.zeros(8, 3)
    weight[3, 0] = 4
    weight[4, 0] = -2
    weight[5, 1] = -2
    weight[6, 1] = 4
    cases = [
        {"prompt_length": 3, "candidate_token_ids": [3, 4], "answer_index": 0},
        {"prompt_length": 2, "candidate_token_ids": [5, 6], "answer_index": 1},
    ]
    assert candidate_ranking_loss(hidden, cases, weight).item() < 0.01


def test_replay_microsteps_are_even_and_exact():
    assert replay_microsteps(4, 0.0) == ()
    assert replay_microsteps(4, 0.5) == (1, 3)
    assert replay_microsteps(4, 0.25) == (3,)
    with pytest.raises(ValueError, match="representable"):
        replay_microsteps(4, 0.3)


def test_training_task_schedule_switches_once():
    values = {
        "tasks": ("two_hop_chain",),
        "after_switch_tasks": ("two_hop_symbolic",),
        "task_switch_step": 10,
    }
    assert training_tasks_for_step(values, 9) == ("two_hop_chain",)
    assert training_tasks_for_step(values, 10) == ("two_hop_symbolic",)


@pytest.mark.parametrize(
    "field,value",
    (
        ("steps", 0),
        ("batch_size", True),
        ("lr", 0.0),
        ("min_lr", 2.0),
        ("train_seed_offset", -1),
    ),
)
def test_settings_reject_invalid_values(field, value):
    with pytest.raises(ValueError):
        validate_settings(settings(**{field: value}))


def test_settings_keep_training_and_validation_loads_separate():
    resolved = validate_settings(
        settings(train_record_counts=(2, 8), validation_record_counts=(2, 8))
    )
    assert resolved["train_record_counts"] == (2, 8)
    assert resolved["validation_record_counts"] == (2, 8)
