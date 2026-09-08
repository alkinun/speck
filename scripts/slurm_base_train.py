"""Run base training with bounded Slurm signal/checkpoint/requeue support."""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import torch

from scripts import base_train
from speck.checkpoint import latest
from speck.common import base_dir
from speck.config import load_experiment
from speck.model import CausalLMTrainingOutput
from speck.slurm import REQUEUE_EXIT_CODE


def arguments(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    if "-h" in raw or "--help" in raw:
        print(
            "Slurm wrapper option:\n"
            "  --slurm-requeue-resume  checkpoint on the timeout signal and resume only retries\n"
        )
    operational = argparse.ArgumentParser(add_help=False)
    operational.add_argument("--slurm-requeue-resume", action="store_true")
    slurm, remaining = operational.parse_known_args(raw)
    cli = base_train.arguments(remaining)
    cli.slurm_requeue_resume = slurm.slurm_requeue_resume
    return cli


def _configure_resume(configs, cli):
    if not cli.slurm_requeue_resume:
        raise ValueError("Slurm training requires --slurm-requeue-resume")
    if cli.resume is not None:
        raise ValueError("--resume and --slurm-requeue-resume are mutually exclusive")
    if "SLURM_JOB_ID" not in os.environ:
        raise ValueError("--slurm-requeue-resume requires a Slurm job")
    try:
        restart_count = int(os.environ.get("SLURM_RESTART_COUNT", "0"))
        retry_offset = int(os.environ.get("SPECK_RETRY_OFFSET", "0"))
    except ValueError as error:
        raise ValueError("Slurm retry counters must be non-negative integers") from error
    if restart_count < 0 or retry_offset < 0:
        raise ValueError("Slurm retry counters must be non-negative integers")
    if not restart_count + retry_offset:
        return
    run = configs["train"].get("run") or Path(cli.experiment).resolve().name
    output_dir = (
        cli.output_dir.expanduser().resolve()
        if cli.output_dir is not None
        else configs["train"].get("output_dir") or Path(base_dir()) / "checkpoints" / run
    )
    cli.resume = latest(output_dir)
    if cli.resume is None:
        raise FileNotFoundError("requeued Slurm job has no complete checkpoint to resume")


class SlurmBaseTrainer(base_train.BaseTrainer):
    """Add a safe optimizer-boundary interruption hook without changing the pinned trainer."""

    def __init__(self, configs, cli):
        self._signal_requested = False
        self.interrupted_for_requeue = False
        _configure_resume(configs, cli)
        super().__init__(configs, cli)

    def _requeue_requested(self):
        signal_file = os.environ.get("SPECK_REQUEUE_SIGNAL_FILE")
        return self._signal_requested or bool(signal_file and Path(signal_file).is_file())

    def _signal_checkpoint(
        self,
        step,
        validation_loss,
        validation_source_losses,
        validation_step,
        validation_tokens,
        milestone,
    ):
        steps = self.steps
        if step == steps:
            self.steps += 1
        try:
            self._checkpoint(
                step,
                validation_loss,
                validation_source_losses,
                validation_step,
                validation_tokens,
                milestone,
            )
        finally:
            self.steps = steps

    def _run_steps(self):
        args = self.args
        validation_loss, validation_source_losses, validation_step, validation_tokens = (
            self._initial_validation()
        )
        synchronize = torch.cuda.synchronize if self.device.type == "cuda" else lambda: None
        timing_started = time.perf_counter()
        timing_steps = 0
        for step in range(self.start_step, self.steps):
            completed = step + 1
            should_diagnose = completed % args.diagnostics_every == 0
            should_log = completed == 1 or completed % args.log_every == 0 or should_diagnose
            milestone = self.milestones.get(completed)
            stop_now = self.stop_step == completed
            should_validate = (
                (args.eval_every > 0 and completed % args.eval_every == 0)
                or milestone is not None
                or completed == self.steps
            )
            should_save = (
                (args.save_every > 0 and completed % args.save_every == 0)
                or milestone is not None
                or completed == self.steps
            )
            scale = base_train.lr_scale(
                self.schedule_step_offset + step,
                self.schedule_steps,
                args.warmup_steps,
                args.min_lr,
                args.lr_schedule,
                args.decay_fraction,
            )
            training_output, grad_norm, batch = base_train.optimization_step(
                self.train_model,
                self.parameters,
                self.optimizer,
                self.train_data,
                (self.inputs, self.targets, self.data_state),
                self.accumulation,
                args.grad_clip,
                args.lr * scale,
                self.distributed,
                return_training_output=True,
                load_balance_coefficient=args.load_balance_coefficient,
                router_z_loss_coefficient=args.router_z_loss_coefficient,
            )
            if not isinstance(training_output, CausalLMTrainingOutput):
                raise TypeError("training step did not return typed loss diagnostics")
            self.inputs, self.targets, self.data_state = batch
            self.completed_step = completed
            timing_steps += 1
            should_flush_timing = should_log or should_validate or should_save or completed == 10
            duration = None
            if should_flush_timing:
                synchronize()
                window_duration = time.perf_counter() - timing_started
                self.elapsed_optimizer += window_duration
                if completed > 10:
                    self.elapsed_training += window_duration
                duration = window_duration / timing_steps
            if self.distributed and should_log:
                base_train.average_training_output(training_output, True)
            if should_log:
                self._log_step(
                    completed,
                    training_output,
                    grad_norm,
                    duration,
                    should_diagnose,
                )
            if self._requeue_requested():
                if not should_flush_timing:
                    synchronize()
                    window_duration = time.perf_counter() - timing_started
                    self.elapsed_optimizer += window_duration
                    if completed > 10:
                        self.elapsed_training += window_duration
                self._signal_checkpoint(
                    completed,
                    validation_loss,
                    validation_source_losses,
                    validation_step,
                    validation_tokens,
                    milestone,
                )
                self.interrupted_for_requeue = True
                break
            if should_validate:
                validation_loss, validation_source_losses, validation_tokens = self._validate(
                    completed
                )
                validation_step = completed
            if should_save:
                self._checkpoint(
                    completed,
                    validation_loss,
                    validation_source_losses,
                    validation_step,
                    validation_tokens,
                    milestone,
                )
            if should_flush_timing:
                timing_started = time.perf_counter()
                timing_steps = 0
            if stop_now:
                break
        if (
            self.start_step == self.steps
            and validation_step != self.steps
            and not self.interrupted_for_requeue
        ):
            validation_loss, validation_source_losses, validation_tokens = self._validate(
                self.steps
            )
            self._checkpoint(
                self.steps,
                validation_loss,
                validation_source_losses,
                self.steps,
                validation_tokens,
                self.milestones.get(self.steps),
            )

    def _write_summary(self):
        super()._write_summary()
        if not self.master or not self.interrupted_for_requeue:
            return
        path = Path(self.args.output_dir) / "run_summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["partial"] = True
        summary["requeue_requested"] = True
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    def run(self):
        previous_handler = signal.getsignal(signal.SIGUSR1)
        signal.signal(
            signal.SIGUSR1, lambda _signum, _frame: setattr(self, "_signal_requested", True)
        )
        try:
            super().run()
            return self.interrupted_for_requeue
        finally:
            signal.signal(signal.SIGUSR1, previous_handler)


def train(configs, cli):
    return SlurmBaseTrainer(configs, cli).run()


def main():
    cli = arguments()
    configs = load_experiment(cli.experiment, "data", "tokenizer", "model", "train")
    if train(configs, cli):
        raise SystemExit(REQUEUE_EXIT_CODE)


if __name__ == "__main__":
    main()
