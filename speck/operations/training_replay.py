"""Bound fresh-process recovery checks through the production training entry points."""

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import torch

from speck.config import load_experiment
from speck.operations.r0_executor import supervise
from speck.provenance.io import atomic_json, file_sha256
from speck.training.checkpoint import latest, load_metadata


def compare_state(expected, actual, *, rtol, atol, path="state"):
    """Compare every tensor and scalar, including optimizer counters and groups."""
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, msg=path)
        return 1
    if type(expected) is not type(actual):
        raise AssertionError(f"{path}: state types differ")
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise AssertionError(f"{path}: state keys differ")
        return sum(
            compare_state(value, actual[key], rtol=rtol, atol=atol, path=f"{path}.{key}")
            for key, value in expected.items()
        )
    if isinstance(expected, (tuple, list)):
        if len(expected) != len(actual):
            raise AssertionError(f"{path}: state lengths differ")
        return sum(
            compare_state(value, actual[i], rtol=rtol, atol=atol, path=f"{path}[{i}]")
            for i, value in enumerate(expected)
        )
    if expected != actual:
        raise AssertionError(f"{path}: scalar state differs")
    return 0


def compare_checkpoints(first, second, step, device):
    tolerance = {"rtol": 0, "atol": 0} if device == "cpu" else {"rtol": 1e-5, "atol": 1e-6}
    result = {"tolerance": tolerance, "tensor_counts": {}}
    for kind in ("model", "optimizer"):
        name = f"{kind}_{step:06d}.pt"
        expected = torch.load(first / name, map_location="cpu", weights_only=True, mmap=True)
        actual = torch.load(second / name, map_location="cpu", weights_only=True, mmap=True)
        result["tensor_counts"][kind] = compare_state(expected, actual, **tolerance, path=kind)
        del expected, actual
    expected, actual = load_metadata(first, step), load_metadata(second, step)
    if "rng_state" not in expected or "data_state" not in expected:
        raise AssertionError("production checkpoint is missing RNG or loader state")
    for key in ("rng_state", "data_state", "step", "trained_tokens", "trained_supervised_tokens"):
        if expected.get(key) != actual.get(key):
            raise AssertionError(f"checkpoint {key} differs after fresh-process restart")
    result["rng_and_loader"] = "exact"
    return result


def replay(args):
    if args.workers < 1 or args.seconds <= 0 or args.checkpoint_step < 1:
        raise ValueError("workers, seconds, and checkpoint step must be positive")
    if args.allocated_gpus < args.workers:
        raise ValueError("allocated GPU count must include every worker and idle allocated GPU")
    if args.steps <= args.checkpoint_step:
        raise ValueError("steps must exceed checkpoint step")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    phase_key = "train" if args.phase == "base" else "sft"
    names = (
        ("data", "tokenizer", "model", "train")
        if args.phase == "base"
        else ("tokenizer", "model", "sft")
    )
    configs = load_experiment(args.experiment, *names)
    settings = configs[phase_key]
    settings.update(
        run="dummy",
        wandb_project=None,
        deterministic=True,
        save_every=args.checkpoint_step,
        eval_every=args.checkpoint_step,
        log_every=1,
    )
    if args.phase == "base":
        batch = settings["device_batch_size"] * settings["sequence_length"] * args.workers
        settings.update(
            batch_tokens=batch,
            train_tokens=args.steps * batch,
            checkpoint_tokens=[args.checkpoint_step * batch],
            warmup_steps=min(settings["warmup_steps"], args.checkpoint_step - 1),
            eval_tokens=batch * len(configs["data"]["sources"]),
            final_eval_tokens=batch * len(configs["data"]["sources"]),
        )
    else:
        from speck.training.sft_data import load_sft_manifest, sft_plan

        tokens = settings["device_batch_size"] * settings["sequence_length"]
        accumulation = settings["batch_tokens"] // (tokens * args.workers)
        if accumulation < 1 or settings["batch_tokens"] % (tokens * args.workers):
            raise ValueError("SFT batch must be divisible by the distributed microbatch")
        plan = sft_plan(
            load_sft_manifest(settings["data_dir"]), "train", tokens, args.workers, accumulation
        )
        steps = plan["cycle_microbatches"] // accumulation * settings["epochs"]
        if steps != args.steps:
            raise ValueError(
                f"SFT experiment has {steps} steps, but the replay bound is {args.steps}"
            )
        settings["keep_checkpoints"] = max(
            settings["keep_checkpoints"], args.steps // args.checkpoint_step + 1
        )
    request = {
        "format": "speck_production_training_replay",
        "format_version": 1,
        "status": "running",
        "phase": args.phase,
        "device": args.device,
        "workers": args.workers,
        "allocated_gpus": args.allocated_gpus,
        "timeout_seconds": args.seconds,
        "source_experiment": str(Path(args.experiment).resolve()),
        "checkpoint_step": args.checkpoint_step,
        "engineering_only": True,
        "compile": False,
        "commands": [],
        "executions": [],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "implementation": [
            {"path": str(path), "sha256": file_sha256(path)}
            for directory in ("speck", "scripts")
            for path in sorted(Path(directory).rglob("*.py"))
        ],
    }
    atomic_json(root / "result.json", request)
    environment_keys = ("MASTER_ADDR", "MASTER_PORT", "OMP_NUM_THREADS", "WANDB_MODE")
    previous = {key: os.environ.get(key) for key in environment_keys}
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), OMP_NUM_THREADS="1", WANDB_MODE="disabled"
    )
    first, second = root / "uninterrupted", root / "resumed"
    try:
        for index, output in enumerate((first, second)):
            directory = root / f"process-{index}"
            directory.mkdir()
            experiment = directory / "experiment"
            experiment.mkdir()
            settings["output_dir"] = str(output)
            for name, config in configs.items():
                atomic_json(experiment / f"{name}.json", config)
            command = [
                sys.executable,
                "-m",
                f"scripts.{args.phase}_train",
                str(experiment),
                "--device",
                args.device,
                "--no-compile",
            ]
            if index:
                if not (first / f"complete_{args.checkpoint_step:06d}").exists():
                    raise AssertionError("intermediate checkpoint was not retained")
                output.mkdir()
                for path in first.glob(f"*_{args.checkpoint_step:06d}*"):
                    if path.is_file() and not path.name.endswith(".tmp"):
                        # Published checkpoint payloads are immutable. Later steps use new names.
                        try:
                            os.link(path, output / path.name)
                        except OSError:
                            shutil.copyfile(path, output / path.name)
                command += ["--resume", str(args.checkpoint_step)]
            remaining = args.seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("production replay exhausted its total time limit")
            request["commands"].append(command)
            atomic_json(root / "result.json", request)
            execution = supervise(command, directory, remaining, 10, args.workers)
            request["executions"].append(execution)
            if execution["returncode"] != 0 or execution["termination"] != "exited":
                raise RuntimeError(f"production trainer failed: {execution}; see {directory}")
        step = latest(first)
        if step is None or step <= args.checkpoint_step or latest(second) != step:
            raise AssertionError("replay did not complete beyond its intermediate checkpoint")
        if step != args.steps:
            raise AssertionError("replay ended at the wrong step")
        request["comparison"] = compare_checkpoints(first, second, step, args.device)
        request["final_step"] = step
        request["metadata_sha256"] = {
            name: file_sha256(path / f"metadata_{step:06d}.json")
            for name, path in (("uninterrupted", first), ("resumed", second))
        }
        request["status"] = "pass"
    except BaseException as error:
        request.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        request["wall_seconds"] = time.monotonic() - started
        request["supervised_gpu_hours"] = (
            sum(row["supervised_wall_seconds"] for row in request["executions"])
            * args.allocated_gpus
            / 3600
            if args.device == "cuda"
            else 0
        )
        request["allocated_gpu_hours"] = (
            request["wall_seconds"] * args.allocated_gpus / 3600 if args.device == "cuda" else 0
        )
        atomic_json(root / "result.json", request)
    return request


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment")
    parser.add_argument("--output", required=True, help="new external artifact directory")
    parser.add_argument("--phase", choices=("base", "sft"), default="base")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--allocated-gpus", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4, help="engineering base horizon")
    parser.add_argument("--checkpoint-step", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=1800)
    print(json.dumps(replay(parser.parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()
