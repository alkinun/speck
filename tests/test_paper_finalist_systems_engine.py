import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from speck.paper_finalist_systems_engine import (
    benchmark_configs,
    execute_windows,
    materialize_paired_batches,
    run_engine,
    run_trial,
    terminal_learning_rate,
    validate_activation,
)
from speck.paper_finalist_systems_workload import build_trial_plan

root = Path(__file__).parents[1]
protocol_path = root / "research" / "paper-1" / "finalist_systems_v2.json"
qualification_path = root / "results" / "Speck-Paper1" / "finalist-qualification-v1.json"
protocol = json.loads(protocol_path.read_text(encoding="utf-8"))


def first_trial(tmp_path):
    return build_trial_plan(protocol_path, qualification_path, tmp_path / "outputs")["trials"][0]


def test_engine_derives_terminal_lr_and_ephemeral_training_config(tmp_path):
    trial = first_trial(tmp_path)
    configs = benchmark_configs(trial, protocol)
    assert terminal_learning_rate(configs["train"]) == pytest.approx(0.00015)
    assert configs["train"]["run"] == "dummy"
    assert configs["train"]["train_tokens"] == 2_621_440
    assert configs["train"]["eval_every"] == 0
    assert configs["train"]["save_every"] == 0
    assert configs["train"]["checkpoint_tokens"] == []


def test_engine_executes_exact_warmup_and_measured_windows():
    steps = []
    synchronizations = []
    resets = []
    markers = {}
    clocks = iter((10.0, 25.0))
    marker_times = iter((1, 2, 3))
    measured = execute_windows(
        lambda: steps.append(len(steps)),
        lambda: synchronizations.append(True),
        lambda: resets.append(True),
        lambda name, value: markers.update({name: value}),
        clock=lambda: next(clocks),
        monotonic_ns=lambda: next(marker_times),
    )
    assert len(steps) == 40
    assert measured == 15
    assert len(synchronizations) == 2
    assert len(resets) == 1
    assert markers == {
        "warmup_start": 1,
        "warmup_end": 2,
        "measured_start": 2,
        "measured_end": 3,
    }


class FakeTrainer:
    def __init__(self, configs, cli):
        self.events = ["construct"]
        self.configs = configs
        self.cli = cli
        self.args = SimpleNamespace(grad_clip=1.0)
        self.device = torch.device("cpu")
        self.distributed = False
        self.train_model = object()
        self.parameters = ()
        self.optimizer = object()
        self.train_data = object()
        self.inputs = 0
        self.targets = 0
        self.data_state = 0
        self.accumulation = 1

    def _initialize_runtime(self):
        self.events.append("runtime")

    def _load_and_verify_data(self):
        self.events.append("data")

    def _initialize_model_and_geometry(self):
        self.events.append("model")

    def _restore_training_state(self):
        self.events.append("restore")

    def _build_resolved_settings(self):
        self.events.append("resolved")

    def _prepare_execution(self):
        self.events.append("prepare")


def test_engine_lifecycle_skips_tracking_validation_checkpoint_and_summary(tmp_path):
    trial = first_trial(tmp_path)
    holder = {}

    def factory(configs, cli):
        holder["trainer"] = FakeTrainer(configs, cli)
        return holder["trainer"]

    step_lrs = []

    def fake_step(model, parameters, optimizer, loader, batch, accumulation, clip, lr, distributed):
        step_lrs.append(lr)
        return torch.tensor(1.0), torch.tensor(1.0), (batch[0] + 1, batch[1], batch[2] + 1)

    cleaned = []
    clocks = iter((0.0, 12.0))
    marker_times = iter(range(10))
    report = run_engine(
        trial,
        protocol,
        tmp_path / "ephemeral",
        trainer_factory=factory,
        optimization_step_fn=fake_step,
        batch_materializer=lambda *_: {
            "sha256": "a" * 64,
            "microbatches": 41,
            "optimizer_steps": 40,
            "accumulation_steps": 1,
            "materialized_device": "cpu",
            "transfer_device": "cpu",
            "H2D_inside_optimizer_windows": True,
        },
        cleanup_fn=lambda: cleaned.append(True),
        device="cpu",
        no_compile=True,
        clock=lambda: next(clocks),
        monotonic_ns=lambda: next(marker_times),
    )
    assert holder["trainer"].events == [
        "construct",
        "runtime",
        "data",
        "model",
        "restore",
        "resolved",
        "prepare",
    ]
    assert len(step_lrs) == 40
    assert step_lrs == pytest.approx([0.00015] * 40)
    assert cleaned == [True]
    assert report["measured_wall_seconds"] == 12
    assert report["validation_executed"] is False
    assert report["checkpoint_write_executed"] is False
    assert report["summary_write_executed"] is False
    assert report["tracking_initialized"] is False
    assert report["model_or_optimizer_output_persisted"] is False
    assert report["kernel_fallback"] is None
    assert report["paired_batches"]["sha256"] == "a" * 64
    assert not (tmp_path / "ephemeral").exists()


def test_engine_cleanup_runs_when_optimization_fails(tmp_path):
    trial = first_trial(tmp_path)
    cleaned = []

    def fail(*_):
        raise FloatingPointError("non-finite")

    with pytest.raises(FloatingPointError, match="non-finite"):
        run_engine(
            trial,
            protocol,
            tmp_path / "ephemeral",
            trainer_factory=FakeTrainer,
            optimization_step_fn=fail,
            batch_materializer=lambda *_: {},
            cleanup_fn=lambda: cleaned.append(True),
            device="cpu",
            no_compile=True,
            monotonic_ns=iter(range(10)).__next__,
        )
    assert cleaned == [True]


def test_engine_rejects_changed_windows():
    with pytest.raises(ValueError, match="windows changed"):
        execute_windows(
            lambda: None,
            lambda: None,
            lambda: None,
            lambda *_: None,
            warmup_steps=9,
        )


def test_engine_activation_absence_fails_before_execution(tmp_path):
    missing = tmp_path / "activation.json"
    with pytest.raises(FileNotFoundError):
        validate_activation(missing, protocol_path)
    output = tmp_path / "block-0-0-control.json"
    with pytest.raises(FileNotFoundError):
        run_trial(
            protocol_path,
            qualification_path,
            missing,
            0,
            0,
            output,
        )
    assert not output.exists()


def test_engine_materializes_exact_CPU_microbatch_replay_and_fingerprint():
    class Trainer:
        tokenizer = object()
        args = SimpleNamespace(device_batch_size=1, sequence_length=2, data_dir="unused")
        accumulation = 2
        device = torch.device("cpu")

    def loader_factory(*_, **kwargs):
        assert kwargs["device"] == "cpu"
        assert kwargs["resume_state_dict"] == {"global_consumed_tokens": 100}
        for index in range(81):
            values = torch.tensor([[index, index + 1]])
            yield values, values + 1, {"global_consumed_tokens": 100 + index * 2}

    first = Trainer()
    identity = materialize_paired_batches(
        first,
        {"global_consumed_tokens": 100},
        40,
        loader_factory=loader_factory,
    )
    second = Trainer()
    repeated = materialize_paired_batches(
        second,
        {"global_consumed_tokens": 100},
        40,
        loader_factory=loader_factory,
    )
    assert identity == repeated
    assert identity["microbatches"] == 81
    assert identity["accumulation_steps"] == 2
    assert identity["materialized_device"] == "cpu"
    assert identity["H2D_inside_optimizer_windows"] is True
    assert next(first.train_data)[0].device.type == "cpu"
