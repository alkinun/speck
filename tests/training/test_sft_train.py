import pytest

from speck.config import load_experiment
from speck.training.sft import _settings
from tests.reference import historical_repository


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
    ),
)
def test_invalid_sft_settings_fail_before_runtime_initialization(field, value):
    experiment = historical_repository() / "experiments" / "Speck1-140M-Instruct"
    config = load_experiment(experiment, "sft")["sft"]
    _settings(config)
    with pytest.raises(ValueError, match="SFT"):
        _settings({**config, field: value})
