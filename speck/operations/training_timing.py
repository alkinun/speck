"""Observe production base training without changing its optimizer or recovery path."""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from speck.config import load_experiment
from speck.training.base import BaseTrainer, arguments


class TimedTrainer(BaseTrainer):
    def __init__(self, configs, cli, events):
        self.events = events
        super().__init__(configs, cli)

    def record(self, event, **values):
        self.events.write(
            json.dumps({"event": event, "utc": datetime.now(timezone.utc).isoformat(), **values})
            + "\n"
        )
        self.events.flush()

    def observe(self, name, call, *args, **kwargs):
        started = time.perf_counter()
        try:
            value = call(*args, **kwargs)
        except BaseException as error:
            self.record(name, seconds=time.perf_counter() - started, error=repr(error))
            raise
        self.record(name, seconds=time.perf_counter() - started)
        return value

    def _restore_training_state(self):
        return self.observe("restore", super()._restore_training_state)

    def _initialize_model_and_geometry(self):
        return self.observe("model_initialization", super()._initialize_model_and_geometry)

    def _load_and_verify_data(self):
        return self.observe("data_verification", super()._load_and_verify_data)

    def _prepare_execution(self):
        return self.observe("prepare_execution", super()._prepare_execution)

    def _validate(self, step):
        started = time.perf_counter()
        result = super()._validate(step)
        self.record(
            "validation", step=step, seconds=time.perf_counter() - started, tokens=result[2]
        )
        return result

    def _checkpoint(self, step, *args, **kwargs):
        started = time.perf_counter()
        result = super()._checkpoint(step, *args, **kwargs)
        seconds = time.perf_counter() - started
        files = list(Path(self.args.output_dir).glob(f"*_{step:06d}*"))
        self.record(
            "checkpoint",
            step=step,
            seconds=seconds,
            bytes=sum(path.stat().st_size for path in files if path.is_file()),
            complete=(Path(self.args.output_dir) / f"complete_{step:06d}").is_file(),
        )
        return result

    def _log_step(self, completed, output, grad_norm, duration, diagnostics):
        result = super()._log_step(completed, output, grad_norm, duration, diagnostics)
        self.record(
            "optimizer_step",
            step=completed,
            seconds=duration,
            tokens=self.args.batch_tokens,
            steady=completed - self.start_step > 10,
        )
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--events", required=True, type=Path)
    wrapper, remaining = parser.parse_known_args(argv)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("this timing observer is single-worker only")
    cli = arguments(remaining)
    configs = load_experiment(cli.experiment, "data", "tokenizer", "model", "train")
    if configs["train"]["log_every"] != 1:
        raise ValueError("timing observations require log_every=1 for per-step durations")
    with wrapper.events.open("x") as events:
        trainer = TimedTrainer(configs, cli, events)
        trainer.observe("session", trainer.run)
