import pytest

from scripts import base_train, sft_train


class TrackingRun:
    def __init__(self, error=None):
        self.error = error
        self.finished = 0

    def finish(self):
        self.finished += 1
        if self.error is not None:
            raise self.error


@pytest.mark.parametrize(
    ("module", "trainer_class", "setup_methods"),
    (
        (
            base_train,
            base_train.BaseTrainer,
            (
                "_initialize_runtime",
                "_load_and_verify_data",
                "_initialize_model_and_geometry",
                "_restore_training_state",
                "_build_resolved_settings",
                "_initialize_tracking",
                "_prepare_execution",
            ),
        ),
        (
            sft_train,
            sft_train.SFTTrainer,
            (
                "_initialize_runtime",
                "_load_and_verify_data",
                "_resolve_geometry",
                "_initialize_model",
                "_build_resolved_settings",
                "_publish_tokenizer",
                "_initialize_tracking",
                "_prepare_execution",
            ),
        ),
    ),
)
def test_training_failure_finishes_tracking_and_cleans_up(
    monkeypatch, module, trainer_class, setup_methods
):
    trainer = object.__new__(trainer_class)
    tracking = TrackingRun()
    trainer.tracking = tracking
    cleaned = []
    monkeypatch.setattr(module, "cleanup", lambda: cleaned.append(True))
    for name in setup_methods:
        monkeypatch.setattr(trainer, name, lambda: None)

    def fail():
        raise RuntimeError("training failed")

    monkeypatch.setattr(trainer, "_run_steps", fail)

    with pytest.raises(RuntimeError, match="training failed"):
        trainer.run()

    assert tracking.finished == 1
    assert cleaned == [True]


@pytest.mark.parametrize(
    ("module", "trainer_class"),
    (
        (base_train, base_train.BaseTrainer),
        (sft_train, sft_train.SFTTrainer),
    ),
)
def test_runtime_initialization_failure_still_cleans_up(monkeypatch, module, trainer_class):
    trainer = object.__new__(trainer_class)
    trainer.tracking = None
    cleaned = []
    monkeypatch.setattr(module, "cleanup", lambda: cleaned.append(True))

    def fail():
        raise RuntimeError("runtime failed")

    monkeypatch.setattr(trainer, "_initialize_runtime", fail)

    with pytest.raises(RuntimeError, match="runtime failed"):
        trainer.run()

    assert cleaned == [True]


def test_tracking_failure_does_not_skip_runtime_cleanup(monkeypatch):
    trainer = object.__new__(base_train.BaseTrainer)
    tracking = TrackingRun(RuntimeError("tracking failed"))
    trainer.tracking = tracking
    cleaned = []
    monkeypatch.setattr(base_train, "cleanup", lambda: cleaned.append(True))
    for name in (
        "_initialize_runtime",
        "_load_and_verify_data",
        "_initialize_model_and_geometry",
        "_restore_training_state",
        "_build_resolved_settings",
        "_initialize_tracking",
        "_prepare_execution",
        "_run_steps",
    ):
        monkeypatch.setattr(trainer, name, lambda: None)

    with pytest.raises(RuntimeError, match="tracking failed"):
        trainer.run()

    assert tracking.finished == 1
    assert cleaned == [True]
