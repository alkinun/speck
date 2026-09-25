import pytest

from speck.training.sft import _settings

# Omits deterministic, activation_checkpointing and loss_backend, as configs written before those
# settings existed do; _settings supplies their defaults.
SFT_CONFIG = {
    "batch_tokens": 65536,
    "data_dir": None,
    "dataset": {
        "expected_samples": 300000,
        "files": ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"],
        "repo": "specklabs/SpeckChat1",
        "revision": "530135f1fcc3ea21eb9ee0f6dd1ab59547a6bee9",
        "validation_samples": 1000,
    },
    "device_batch_size": 4,
    "epochs": 1,
    "eval_every": 100,
    "grad_clip": 1.0,
    "keep_checkpoints": 2,
    "log_every": 5,
    "lr": 0.0001,
    "min_lr": 0.1,
    "optimizer": "adamw",
    "output_dir": None,
    "pretrained": {
        "filename": "model.safetensors",
        "repo": "specklabs/Speck1-140M",
        "revision": "32675011a75e3bb3f180983a0014de10d1fa6693",
    },
    "run": "Speck1-140M-Instruct",
    "save_every": 500,
    "sequence_length": 2048,
    "sequence_lengths": [256, 512, 1024, 2048],
    "wandb_project": "speck",
    "warmup_steps": 100,
    "weight_decay": 0.1,
}


@pytest.mark.parametrize(
    "field,value",
    (
        ("sequence_lengths", []),
        ("sequence_lengths", [[4096]]),
        ("sequence_lengths", [0, 4096]),
        ("sequence_lengths", [True, 4096]),
        ("epochs", True),
        ("eval_every", 0.5),
        ("save_every", None),
        ("lr", float("nan")),
        ("weight_decay", float("inf")),
        ("grad_clip", "1"),
        ("min_lr", True),
        ("deterministic", "true"),
    ),
)
def test_invalid_sft_settings_fail_before_runtime_initialization(field, value):
    _settings(SFT_CONFIG)
    with pytest.raises(ValueError, match="SFT"):
        _settings({**SFT_CONFIG, field: value})
