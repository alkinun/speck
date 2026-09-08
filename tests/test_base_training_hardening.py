import json
import time
from types import SimpleNamespace

import pytest
import torch

from scripts import base_train, slurm_base_train
from speck.checkpoint import save
from speck.model import CausalLMTrainingOutput


def _summary_trainer(trainer_class, output_dir, completed, steps):
    trainer = object.__new__(trainer_class)
    trainer.master = True
    trainer.completed_step = completed
    trainer.steps = steps
    trainer.global_step_offset = 0
    trainer.stop_step = completed if completed < steps else None
    trainer.parameters = (torch.nn.Parameter(torch.tensor(1.0)),)
    trainer.distributed = False
    trainer.validation_history = [{"step": completed}]
    trainer.args = SimpleNamespace(
        training_phase="base",
        global_token_offset=0,
        batch_tokens=8,
        stop_at_tokens=completed * 8 if completed < steps else None,
        output_dir=str(output_dir),
    )
    trainer.elapsed_optimizer = 3.0
    trainer.elapsed_training = 2.0
    trainer.elapsed_evaluation = 0.5
    trainer.elapsed_checkpoint = 0.25
    trainer.elapsed_active = 0.0
    trainer.session_started = time.perf_counter()
    trainer.device = torch.device("cpu")
    return trainer


def test_milestone_summary_has_a_distinct_nonfinal_contract(tmp_path):
    trainer = _summary_trainer(base_train.BaseTrainer, tmp_path, completed=2, steps=3)
    trainer._write_summary()

    assert not (tmp_path / "run_summary.json").exists()
    partial = json.loads((tmp_path / "partial_run_summary_000002.json").read_text())
    assert partial["format"] == "speck_base_partial_run_summary"
    assert partial["status"] == "token_milestone"
    assert partial["partial"] is partial["non_final"] is True


def test_partial_summary_preserves_and_retires_legacy_canonical_partial(tmp_path):
    legacy = {"partial": True, "completed_steps": 1, "historical": "preserve"}
    (tmp_path / "run_summary.json").write_text(json.dumps(legacy))
    trainer = _summary_trainer(base_train.BaseTrainer, tmp_path, completed=2, steps=3)

    trainer._write_summary()

    assert not (tmp_path / "run_summary.json").exists()
    assert json.loads((tmp_path / "legacy_partial_run_summary_000001.json").read_text()) == legacy


def test_requeue_summary_cannot_be_confused_with_completion(tmp_path):
    trainer = _summary_trainer(slurm_base_train.SlurmBaseTrainer, tmp_path, completed=3, steps=3)
    trainer.interrupted_for_requeue = True
    trainer._write_summary()

    assert not (tmp_path / "run_summary.json").exists()
    partial = json.loads((tmp_path / "partial_run_summary_000003.json").read_text())
    assert partial["status"] == "requeue_requested"
    assert partial["partial"] is True


def test_completed_summary_rejects_nonfinite_parameters(tmp_path):
    trainer = _summary_trainer(base_train.BaseTrainer, tmp_path, completed=3, steps=3)
    trainer.parameters[0].data.fill_(float("nan"))

    with pytest.raises(FloatingPointError, match="model parameters"):
        trainer._write_summary()
    assert not (tmp_path / "run_summary.json").exists()


def test_completed_checkpoint_rejects_nonfinite_parameters_before_save(tmp_path):
    trainer = object.__new__(base_train.BaseTrainer)
    trainer.steps = 2
    trainer.parameters = (torch.nn.Parameter(torch.tensor(float("inf"))),)
    trainer.distributed = False
    trainer.master = False
    trainer.device = torch.device("cpu")
    trainer.args = SimpleNamespace(output_dir=str(tmp_path))

    with pytest.raises(FloatingPointError, match="model parameters"):
        trainer._checkpoint(2, 1.0, {}, 2, 8, None)


@pytest.mark.parametrize(
    ("loss", "source_losses"),
    ((float("nan"), {}), (1.0, {"web": float("inf")})),
)
def test_base_validation_rejects_nonfinite_before_history_or_tracking(
    tmp_path, monkeypatch, loss, source_losses
):
    trainer = object.__new__(base_train.BaseTrainer)
    trainer.args = SimpleNamespace(
        device_batch_size=1,
        sequence_length=8,
        final_eval_tokens=8,
        eval_tokens=8,
        data_dir=str(tmp_path),
        global_token_offset=0,
        batch_tokens=8,
    )
    trainer.steps = 1
    trainer.world_size = 1
    trainer.manifest = {"splits": {"val": {"tokens": 8}}}
    trainer.tokenizer = object()
    trainer.device = torch.device("cpu")
    trainer.train_model = object()
    trainer.source_ids = ("web",)
    trainer.elapsed_evaluation = 0.0
    trainer.global_step_offset = 0
    trainer.validation_history = []
    logged = []
    trainer.tracking = SimpleNamespace(log=lambda metrics: logged.append(metrics))
    monkeypatch.setattr(base_train, "packed_loader", lambda *args, **kwargs: object())
    monkeypatch.setattr(base_train, "validate", lambda *args, **kwargs: (loss, source_losses))

    with pytest.raises(FloatingPointError, match="base validation"):
        trainer._validate(1)
    assert trainer.validation_history == []
    assert logged == []


def test_checkpoint_json_refuses_nan(tmp_path):
    with pytest.raises(ValueError, match="JSON compliant"):
        save(tmp_path, 1, {}, {}, {"step": 1, "validation_loss": float("nan")})
    assert not (tmp_path / "complete_000001").exists()


def test_resume_compile_window_is_counted_as_startup_not_steady(monkeypatch):
    trainer = object.__new__(base_train.BaseTrainer)
    trainer.args = SimpleNamespace(
        diagnostics_every=1_000,
        log_every=1_000,
        eval_every=0,
        save_every=0,
        warmup_steps=0,
        min_lr=1.0,
        lr_schedule="constant",
        decay_fraction=None,
        grad_clip=1.0,
        lr=1e-3,
        load_balance_coefficient=0.0,
        router_z_loss_coefficient=0.0,
    )
    trainer.start_step = trainer.completed_step = 20
    trainer.steps = trainer.schedule_steps = 31
    trainer.schedule_step_offset = 0
    trainer.train_model = object()
    trainer.parameters = ()
    trainer.optimizer = object()
    trainer.train_data = iter(())
    trainer.inputs = trainer.targets = trainer.data_state = object()
    trainer.accumulation = 1
    trainer.distributed = False
    trainer.device = torch.device("cpu")
    trainer.milestones = {}
    trainer.stop_step = None
    trainer.elapsed_optimizer = trainer.elapsed_training = 0.0
    trainer._initial_validation = lambda: (1.0, {}, 20, 8)
    trainer._validate = lambda step: (0.9, {}, 8)
    trainer._checkpoint = lambda *args, **kwargs: None

    def optimization(*args, **kwargs):
        zero = torch.tensor(0.0)
        return CausalLMTrainingOutput(zero, zero, zero, zero, ()), zero, (object(),) * 3

    clock = iter((0.0, 100.0, 100.0, 110.0, 110.0))
    monkeypatch.setattr(base_train, "optimization_step", optimization)
    monkeypatch.setattr(base_train.time, "perf_counter", lambda: next(clock))

    trainer._run_steps()

    assert trainer.elapsed_optimizer == 110.0
    assert trainer.elapsed_training == 10.0
