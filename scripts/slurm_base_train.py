"""Run base training with bounded Slurm signal/checkpoint/requeue support."""

import argparse
import os
import signal
import sys
from pathlib import Path

import torch

from scripts import base_train
from speck.checkpoint import latest
from speck.common import base_dir
from speck.config import load_experiment
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
        max_retries = int(os.environ["SPECK_MAX_RETRIES"])
    except (KeyError, ValueError) as error:
        raise ValueError("Slurm retry counters must be non-negative integers") from error
    if restart_count < 0 or retry_offset < 0 or max_retries < 0:
        raise ValueError("Slurm retry counters must be non-negative integers")
    if restart_count > max_retries or retry_offset > max_retries:
        raise ValueError("Slurm retry bound is exhausted")
    if not max(restart_count, retry_offset):
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
        requested = self._signal_requested or bool(signal_file and Path(signal_file).is_file())
        if self.distributed:
            shared = torch.tensor(int(requested), dtype=torch.int32, device=self.device)
            torch.distributed.all_reduce(shared, op=torch.distributed.ReduceOp.MAX)
            requested = bool(shared.item())
        return requested

    def _optimizer_boundary_stop_requested(self):
        return self._requeue_requested()

    def _after_optimizer_step(
        self,
        step,
        validation_loss,
        validation_source_losses,
        validation_step,
        validation_tokens,
        milestone,
        stop_requested,
    ):
        if not stop_requested:
            return False
        self._checkpoint(
            step,
            validation_loss,
            validation_source_losses,
            validation_step,
            validation_tokens,
            milestone,
            partial=True,
        )
        self.interrupted_for_requeue = True
        return True

    def _nonfinal_summary_reason(self):
        if self.interrupted_for_requeue:
            return "requeue_requested"
        return super()._nonfinal_summary_reason()

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
