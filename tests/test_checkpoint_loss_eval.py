import pytest

from scripts.checkpoint_loss_eval import _positive_integer, arguments


def test_checkpoint_loss_arguments_select_model_and_data_independently():
    args = arguments(
        [
            "model",
            "--data-experiment",
            "data",
            "--sequence-length",
            "4096",
            "--batch-size",
            "4",
            "--loss-backend",
            "liger",
            "--rope-scaling-factor",
            "1",
        ]
    )
    assert str(args.model_experiment) == "model"
    assert str(args.data_experiment) == "data"
    assert args.sequence_length == 4_096
    assert args.batch_size == 4
    assert args.loss_backend == "liger"
    assert args.rope_scaling_factor == 1.0


@pytest.mark.parametrize("value", (0, -1, True, 1.5))
def test_checkpoint_loss_requires_positive_integer_geometry(value):
    with pytest.raises(ValueError, match="positive integer"):
        _positive_integer(value, "value")
