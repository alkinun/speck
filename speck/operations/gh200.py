"""Bundle portable inputs and run the bounded one-GPU GH200 check."""

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace

from speck.config import load_experiment
from speck.export.pretrained import native_pretrained_source
from speck.operations.supervise import supervise
from speck.operations.training_replay import replay
from speck.provenance.io import atomic_json, file_sha256


def bundle(output, data, tokenizer, assistant):
    """Copy only explicit prepared inputs plus committed Git objects; never local environments."""
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("commit or resolve working-tree changes before building a bundle")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    branch = subprocess.check_output(["git", "symbolic-ref", "--short", "HEAD"], text=True).strip()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(
        ["git", "bundle", "create", str(output / "code.bundle"), f"refs/heads/{branch}", "--tags"],
        check=True,
    )
    for source, name in ((data, "pilot-data"), (tokenizer, "tokenizer"), (assistant, "assistant")):
        source = Path(source).resolve()
        if not source.is_dir():
            raise ValueError(f"missing prepared input: {source}")
        target = output / name
        target.mkdir()
        # Packed inputs and tokenizer files only; receipts retain their upstream identities.
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.suffix in {
                ".json",
                ".jsonl",
                ".model",
                ".bin",
                ".idx",
                ".npy",
                ".npz",
                ".parquet",
            }:
                if path.is_symlink():
                    raise ValueError("bundle inputs must be regular files")
                relative = path.relative_to(source)
                destination = target / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
    files = [
        {
            "path": str(path.relative_to(output)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file()
    ]
    manifest = {
        "format": "speck_gh200_bundle",
        "format_version": 1,
        "commit": commit,
        "branch": branch,
        "files": files,
        "payload_bytes": sum(row["bytes"] for row in files),
        "purpose": "Private engineering check; no corpus redistribution or training launch.",
    }
    atomic_json(output / "bundle.json", manifest)
    archive = output.with_suffix(".tar.gz")
    if archive.exists():
        raise FileExistsError(archive)
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(output, arcname=output.name)
    atomic_json(
        output.with_suffix(".transfer.json"),
        {
            "archive": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": file_sha256(archive),
            "commit": commit,
        },
    )
    return manifest


def bind(root):
    """Verify transported bytes and create explicitly relocated engineering experiments."""
    root = Path(root).resolve()
    manifest = json.loads((root / "bundle.json").read_text())
    if manifest.get("format") != "speck_gh200_bundle" or manifest.get("format_version") != 1:
        raise ValueError("unsupported bundle")
    for row in manifest["files"]:
        path = root / row["path"]
        if not path.resolve().is_relative_to(root) or file_sha256(path) != row["sha256"]:
            raise ValueError(f"bundle input identity mismatch: {row['path']}")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if (
        commit != manifest["commit"]
        or subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    ):
        raise ValueError("run from the clean bundled checkout at its recorded commit")
    configs = load_experiment("experiments/pilot", "model", "tokenizer", "data", "train")
    configs["tokenizer"] = {"directory": str(root / "tokenizer")}
    configs["data"].pop("output_name", None)
    configs["data"]["output_dir"] = str(root / "pilot-data")
    directory = root / "relocated-base"
    directory.mkdir(exist_ok=True)
    for name, config in configs.items():
        atomic_json(directory / f"{name}.json", config)
    return manifest, configs, directory


def assistant_experiment(root, configs, parent):
    from speck.training.sft_data import load_sft_manifest, sft_plan

    packed = root / "assistant/packed"
    manifest = load_sft_manifest(packed)
    plan = sft_plan(manifest, "train", 4096, 1, 1)
    microbatches = plan["cycle_microbatches"]
    if microbatches < 4 or microbatches % 4:
        raise ValueError("assistant rehearsal must contain a multiple of four microbatches")
    settings = {
        "deterministic": True,
        "activation_checkpointing": True,
        "loss_backend": "liger",
        "batch_tokens": microbatches // 4 * 4096,
        "device_batch_size": 1,
        "sequence_length": 4096,
        "sequence_lengths": [4096],
        "epochs": 1,
        "dataset": json.loads((root / "assistant/dataset.json").read_text()),
        "data_dir": str(packed),
        "eval_every": 2,
        "save_every": 2,
        "log_every": 1,
        "keep_checkpoints": 3,
        "grad_clip": 1.0,
        "lr": 1e-5,
        "min_lr": 0.1,
        "optimizer": "muon",
        "run": "dummy",
        "wandb_project": None,
        "warmup_steps": 0,
        "weight_decay": 0.1,
        "output_dir": None,
        "pretrained": native_pretrained_source(parent, 4),
    }
    experiment = root / "relocated-sft"
    experiment.mkdir(exist_ok=True)
    for name, config in {
        "sft": settings,
        "model": configs["model"],
        "tokenizer": configs["tokenizer"],
    }.items():
        atomic_json(experiment / f"{name}.json", config)
    return experiment


def export_arguments(phase, checkpoint, output, tokenizer):
    """Select the phase-specific exporter and keep artifacts local."""
    if phase == "base":
        return [
            "-m",
            "scripts.base_checkpoint_export",
            str(checkpoint),
            "--step",
            "4",
            "--output-dir",
            str(output),
            "--tokenizer-dir",
            str(tokenizer),
        ]
    if phase == "sft":
        return [
            "-m",
            "scripts.model_publish",
            "--checkpoint-dir",
            str(checkpoint),
            "--step",
            "4",
            "--output-dir",
            str(output),
            "--repo",
            "local/gh200-check",
            "--expected-epochs",
            "1",
            "--no-upload",
        ]
    raise ValueError(f"unsupported export phase: {phase}")


def run(root, output, seconds, *, allow_other_gpu=False):
    import torch

    if not 60 <= seconds <= 7200:
        raise ValueError("check time limit must be between 60 and 7200 seconds")
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {"status": "running", "phases": {}, "timeout_seconds": seconds, "allocated_gpus": 1}
    atomic_json(output / "result.json", result)
    try:
        manifest, configs, experiment = bind(root)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("this check requires exactly one visible allocated GPU")
        device = torch.cuda.get_device_properties(0)
        if not allow_other_gpu and (platform.machine() != "aarch64" or "GH200" not in device.name):
            raise ValueError(
                "expected an ARM64 GH200; --allow-other-gpu permits other engineering hardware"
            )
        if shutil.disk_usage(output).free < 128 * 1024**3:
            raise ValueError("at least 128 GiB of free checkpoint storage is required")
        result["environment"] = {
            "machine": platform.machine(),
            "python": sys.version,
            "gpu": device.name,
            "vram_bytes": device.total_memory,
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
            "commit": manifest["commit"],
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("triton", "flash-linear-attention", "liger-kernel", "numpy", "pyarrow")
            },
            "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True),
        }
        os.environ.update(
            PYTORCH_ALLOC_CONF="expandable_segments:True",
            OMP_NUM_THREADS="1",
            PYTHONUNBUFFERED="1",
            WANDB_MODE="disabled",
        )

        def remaining():
            value = seconds - (time.monotonic() - started)
            if value <= 0:
                raise TimeoutError("check exhausted its time limit")
            return value

        def command(name, arguments):
            directory = output / name
            directory.mkdir()
            execution = supervise([sys.executable, *arguments], directory, remaining(), 20)
            result["phases"][name] = execution
            atomic_json(output / "result.json", result)
            if execution["returncode"] != 0 or execution["termination"] != "exited":
                raise RuntimeError(f"{name} failed; inspect {directory / 'worker.log'}")

        command(
            "loader-scan",
            [
                "-m",
                "scripts.loader_check",
                str(experiment),
                "--batches",
                "64",
                "--output-dir",
                str(output / "loader"),
            ],
        )
        command(
            "loader-replay",
            [
                "-m",
                "scripts.loader_check",
                str(experiment),
                "--batches",
                "64",
                "--mode",
                "replay",
                "--output-dir",
                str(output / "loader"),
            ],
        )
        command(
            "kernels",
            [
                "-m",
                "scripts.kda_kernel_qualify",
                "--lengths",
                "64,256,4096",
                "--gradient-length",
                "64",
                "--decode-length",
                "64",
                "--output",
                str(output / "kernels/result.json"),
            ],
        )
        for phase, config in (("base", experiment), ("sft", None)):
            if phase == "sft":
                config = assistant_experiment(root, configs, output / "base/uninterrupted")
            result["phases"][phase] = replay(
                SimpleNamespace(
                    experiment=str(config),
                    output=str(output / phase),
                    phase=phase,
                    device="cuda",
                    workers=1,
                    allocated_gpus=1,
                    seconds=remaining(),
                    steps=4,
                    checkpoint_step=2,
                )
            )
            atomic_json(output / "result.json", result)
            checkpoint = output / phase / "uninterrupted"
            command(
                f"{phase}-inference",
                [
                    "-m",
                    "scripts.infer",
                    "Explain what a prime number is.",
                    "--experiment",
                    str(config),
                    "--checkpoint-dir",
                    str(checkpoint),
                    "--step",
                    "4",
                    "--max-tokens",
                    "16",
                    "--temperature",
                    "0",
                    "--device",
                    "cuda",
                ],
            )
            command(
                f"{phase}-export",
                export_arguments(
                    phase, checkpoint, output / f"{phase}-exported", root / "tokenizer"
                ),
            )
        result["status"] = "pass"
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["wall_seconds"] = time.monotonic() - started
        result["allocated_gpu_hours"] = result["wall_seconds"] / 3600
        result["boundary"] = (
            "One GPU, eager execution, engineering data only. Installation, transfer, idle and storage charges and earlier attempts are outside this run record. Four-worker, compiled, scheduler, and production-pilot qualification remain separate."
        )
        atomic_json(output / "result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("bundle")
    build.add_argument("--output", required=True)
    build.add_argument("--data", required=True)
    build.add_argument("--tokenizer", required=True)
    build.add_argument("--assistant", required=True)
    bind_parser = commands.add_parser("bind")
    bind_parser.add_argument("root")
    execute = commands.add_parser("run")
    execute.add_argument("root")
    execute.add_argument("--output", required=True)
    execute.add_argument("--seconds", type=int, default=5400)
    execute.add_argument("--allow-other-gpu", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "bundle":
        result = bundle(args.output, args.data, args.tokenizer, args.assistant)
    elif args.command == "bind":
        manifest, _, directory = bind(args.root)
        result = {"commit": manifest["commit"], "experiment": str(directory)}
    else:
        result = run(args.root, args.output, args.seconds, allow_other_gpu=args.allow_other_gpu)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
