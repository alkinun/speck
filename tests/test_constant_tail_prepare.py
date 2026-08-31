import pytest

from scripts.constant_tail_prepare import constant_tail_config, next_learning_rate, parse_args
from speck.train import lr_scale


def parent_metadata():
    return {
        "step": 30,
        "resolved": {
            "batch_tokens": 10,
            "decay_steps": None,
            "device_batch_size": 2,
            "grad_clip": 1.0,
            "loss_backend": "torch",
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


def test_constant_tail_uses_parent_next_learning_rate_and_fixed_recipe():
    metadata = parent_metadata()
    parent = {
        key: value
        for key, value in metadata["resolved"].items()
        if key not in {"schedule_steps", "steps", "world_size"}
    }
    parent.update(
        {
            "checkpoint_tokens": [500],
            "eval_every": 2,
            "output_dir": "/old",
            "run": "parent",
            "save_every": 3,
            "train_tokens": 1000,
        }
    )

    train = constant_tail_config(parent, metadata, train_tokens=200, run="tail")

    expected = 1e-3 * lr_scale(30, 100, 10, 0.1)
    assert train["lr"] == pytest.approx(expected)
    assert train["lr_schedule"] == "constant"
    assert train["warmup_steps"] == 0
    assert train["min_lr"] == 1.0
    assert train["train_tokens"] == 200
    assert train["run"] == "tail"
    assert train["output_dir"] is None
    assert train["checkpoint_tokens"] == []
    assert train["device_batch_size"] == 2
    assert "decay_steps" not in train


def test_constant_tail_rejects_exhausted_schedule_or_recipe_drift():
    metadata = parent_metadata()
    metadata["step"] = 100
    with pytest.raises(ValueError, match="exhausted"):
        next_learning_rate(metadata)

    metadata = parent_metadata()
    parent = {**metadata["resolved"], "weight_decay": 0.2}
    with pytest.raises(ValueError, match="weight_decay"):
        constant_tail_config(parent, metadata, 100, "tail")


def test_constant_tail_arguments_are_explicit():
    args = parse_args(
        [
            "parent",
            "tail",
            "--checkpoint-dir",
            "checkpoints",
            "--step",
            "30",
            "--train-tokens",
            "100",
        ]
    )
    assert args.step == 30
    assert args.train_tokens == 100
