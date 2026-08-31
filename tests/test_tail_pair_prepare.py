import pytest

from scripts.tail_pair_prepare import next_learning_rate, parse_args, tail_configs
from speck.train import lr_scale


def parent_metadata():
    return {
        "step": 30,
        "resolved": {
            "batch_tokens": 10,
            "decay_steps": None,
            "device_batch_size": 2,
            "grad_clip": 1.0,
            "lr": 1e-3,
            "lr_schedule": "cosine",
            "min_lr": 0.1,
            "optimizer": "muon",
            "schedule_steps": 100,
            "sequence_length": 5,
            "steps": 100,
            "warmup_steps": 10,
            "weight_decay": 0.1,
            "world_size": 1,
        },
    }


def parent_train(metadata):
    train = {
        key: value
        for key, value in metadata["resolved"].items()
        if key not in {"schedule_steps", "steps", "world_size"}
    }
    train.update(
        {
            "checkpoint_tokens": [500],
            "eval_every": 2,
            "output_dir": "/old",
            "run": "parent",
            "save_every": 3,
            "train_tokens": 1000,
        }
    )
    return train


def test_tail_pair_matches_budget_cadence_and_fixed_recipe():
    metadata = parent_metadata()
    control, constant = tail_configs(
        parent_train(metadata),
        metadata,
        train_tokens=200,
        run_prefix="tail",
        save_every=5,
        eval_every=4,
    )

    expected = 1e-3 * lr_scale(30, 100, 10, 0.1)
    assert control["lr"] == 1e-3
    assert control["lr_schedule"] == "cosine"
    assert constant["lr"] == pytest.approx(expected)
    assert constant["lr_schedule"] == "constant"
    assert constant["warmup_steps"] == 0
    assert constant["min_lr"] == 1.0
    for key in ("train_tokens", "device_batch_size", "save_every", "eval_every"):
        assert control[key] == constant[key]
    assert control["run"] == "tail-Control"
    assert constant["run"] == "tail-Constant"
    assert control["checkpoint_tokens"] == constant["checkpoint_tokens"] == []
    assert control["output_dir"] is constant["output_dir"] is None
    assert "decay_steps" not in control and "decay_steps" not in constant


def test_tail_pair_rejects_exhausted_schedule_or_recipe_drift():
    metadata = parent_metadata()
    metadata["step"] = 100
    with pytest.raises(ValueError, match="exhausted"):
        next_learning_rate(metadata)

    metadata = parent_metadata()
    parent = {**parent_train(metadata), "weight_decay": 0.2}
    with pytest.raises(ValueError, match="weight_decay"):
        tail_configs(parent, metadata, 100, "tail")

    metadata = parent_metadata()
    metadata["step"] = 95
    with pytest.raises(ValueError, match="exceeds"):
        tail_configs(parent_train(metadata), metadata, 60, "tail")


def test_tail_pair_arguments_are_explicit():
    args = parse_args(
        [
            "parent",
            "pair",
            "--checkpoint-dir",
            "checkpoints",
            "--step",
            "30",
            "--train-tokens",
            "100",
            "--save-every",
            "5",
        ]
    )
    assert args.step == 30
    assert args.train_tokens == 100
    assert args.save_every == 5
