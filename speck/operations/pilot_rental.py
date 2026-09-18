"""Package and supervise the frozen pilot on one H100, with preserved budget reservations."""

import argparse
import fcntl
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import time
import uuid
from pathlib import Path

from speck.config import load_experiment
from speck.operations.r0_executor import fingerprint, supervise
from speck.provenance.io import durable_json, file_sha256

FORMAT = "speck_h100_pilot_packet_v1"
SESSION_SECONDS = 6 * 3600
CEILING_HOURS = 50
GRACE_SECONDS = 15
ENVIRONMENT = {
    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    "OMP_NUM_THREADS": "1",
    "PYTHONUNBUFFERED": "1",
    "WANDB_MODE": "disabled",
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONOPTIMIZE": "0",
}


def clean_commit():
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("use a clean committed checkout")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def copy_tree(source, target):
    source, target = Path(source), Path(target)
    if not source.is_dir():
        raise ValueError(f"missing input directory: {source}")
    if any(path.is_symlink() for path in source.rglob("*")):
        raise ValueError(f"input tree contains symlinks: {source}")
    shutil.copytree(source, target)


def prepare(output, data, tokenizer, evaluation, nltk_data):
    """Build a private relocatable archive; never allocate a GPU or launch training."""
    from huggingface_hub import snapshot_download

    from speck.data.dataset import verify_shards
    from speck.export.checkpoint import TEMPLATE_FILES, TEMPLATE_REPO, TEMPLATE_REVISION
    from speck.tokenization.tokenizer import Tokenizer

    commit = clean_commit()
    receipt = json.loads(Path("experiments/pilot/preparation.json").read_text())
    for name, expected in receipt["configs"].items():
        if file_sha256(Path("experiments/pilot") / name) != expected:
            raise ValueError(f"frozen pilot configuration changed: {name}")
    for path, key in (
        (Path(data) / "manifest.json", "packed_manifest"),
        (evaluation, "evaluation_partitions"),
    ):
        if file_sha256(path) != receipt["artifacts"][key]["sha256"]:
            raise ValueError(f"input differs from frozen pilot preparation: {key}")
    verify_shards(data)
    configs = load_experiment("experiments/pilot", "model", "train", "tokenizer", "data")
    expected_tokenizer = Tokenizer(Path(configs["tokenizer"]["directory"]) / "tokenizer.model")
    if (
        Tokenizer(Path(tokenizer) / "tokenizer.model").fingerprint()
        != expected_tokenizer.fingerprint()
    ):
        raise ValueError("tokenizer differs from frozen pilot")
    prepared = json.loads(Path(evaluation).read_text())
    protocol = Path("experiments/pilot/evaluation.json")
    if prepared["protocol_sha256"] != file_sha256(protocol):
        raise ValueError("evaluation protocol mismatch")
    template = Path(
        snapshot_download(
            TEMPLATE_REPO,
            revision=TEMPLATE_REVISION,
            allow_patterns=list(TEMPLATE_FILES),
            local_files_only=True,
        )
    )
    output = Path(output).resolve()
    archive = output.with_suffix(".tar.gz")
    if archive.exists():
        raise FileExistsError(archive)
    output.mkdir(parents=True, exist_ok=False)
    copy_tree(data, output / "data")
    copy_tree(tokenizer, output / "tokenizer")
    copy_tree(
        Path(nltk_data) / "tokenizers/punkt_tab/english",
        output / "nltk/tokenizers/punkt_tab/english",
    )
    cache = (
        output
        / "hub"
        / ("models--" + TEMPLATE_REPO.replace("/", "--"))
        / "snapshots"
        / TEMPLATE_REVISION
    )
    cache.mkdir(parents=True)
    for name in TEMPLATE_FILES:
        shutil.copyfile(template / name, cache / name)
    (output / "evaluation").mkdir()
    for row in prepared["benchmarks"]:
        source = Path(row["path"])
        if file_sha256(source) != row["sha256"]:
            raise ValueError(f"benchmark changed: {row['id']}")
        target = output / "evaluation" / (row["id"] + "." + row["format"])
        shutil.copyfile(source, target)
        row["path"] = str(target.relative_to(output))
    durable_json(output / "prepared.json", prepared)
    shutil.copyfile(protocol, output / "protocol.json")
    durable_json(output / "configs.json", configs)
    branch = subprocess.check_output(["git", "symbolic-ref", "--short", "HEAD"], text=True).strip()
    subprocess.run(
        ["git", "bundle", "create", str(output / "code.bundle"), f"refs/heads/{branch}"], check=True
    )
    files = [
        {"path": str(p.relative_to(output)), "bytes": p.stat().st_size, "sha256": file_sha256(p)}
        for p in sorted(output.rglob("*"))
        if p.is_file()
    ]
    manifest = {
        "format": FORMAT,
        "commit": commit,
        "branch": branch,
        "files": files,
        "session_seconds": SESSION_SECONDS,
        "pilot_ceiling_gpu_hours": CEILING_HOURS,
        "allocated_gpus": 1,
        "purpose": "Private frozen 105M pilot and development evaluation only; no corpus redistribution.",
    }
    manifest["sha256"] = fingerprint(manifest)
    durable_json(output / "packet.json", manifest)
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(output, arcname=output.name)
    durable_json(
        output.with_suffix(".transfer.json"),
        {
            "archive": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": file_sha256(archive),
            "commit": commit,
        },
    )
    return manifest


def verify_packet(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "packet.json").read_text())
    expected = manifest.pop("sha256")
    if manifest.get("format") != FORMAT or fingerprint(manifest) != expected:
        raise ValueError("invalid pilot packet identity")
    if manifest["commit"] != clean_commit():
        raise ValueError("checkout differs from packet commit")
    for row in manifest["files"]:
        path = root / row["path"]
        if (
            not path.resolve().is_relative_to(root)
            or path.stat().st_size != row["bytes"]
            or file_sha256(path) != row["sha256"]
        ):
            raise ValueError(f"pilot packet input changed: {row['path']}")
    manifest["sha256"] = expected
    return manifest


def bind(root, output):
    """Write machine-specific paths outside the frozen packet and source checkout."""
    root, output = Path(root).resolve(), Path(output).resolve()
    configs = json.loads((root / "configs.json").read_text())
    configs["tokenizer"] = {"directory": str(root / "tokenizer")}
    configs["data"].pop("output_name", None)
    configs["data"]["output_dir"] = str(root / "data")
    configs["train"]["wandb_project"] = None
    for name, value in configs.items():
        durable_json(output / "experiment" / f"{name}.json", value)
    prepared = json.loads((root / "prepared.json").read_text())
    for row in prepared["benchmarks"]:
        row["path"] = str(root / row["path"])
    durable_json(output / "prepared.json", prepared)


def commands(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    evaluation = [str(root / "protocol.json"), str(output / "prepared.json")]
    return [
        (
            "preflight",
            [
                sys.executable,
                "-m",
                "scripts.pilot_rental",
                "preflight",
                str(root),
                "--output",
                str(output),
            ],
        ),
        (
            "graders",
            [
                sys.executable,
                "-m",
                "scripts.capability_eval",
                *evaluation,
                "--qualify",
                "--output",
                str(output / "grader-check"),
            ],
        ),
        (
            "train",
            [
                sys.executable,
                "-m",
                "scripts.base_train",
                str(output / "experiment"),
                "--device",
                "cuda",
                "--no-compile",
                "--output-dir",
                str(output / "checkpoints"),
            ],
        ),
        (
            "export",
            [
                sys.executable,
                "-m",
                "scripts.base_checkpoint_export",
                str(output / "checkpoints"),
                "--step",
                "800",
                "--output-dir",
                str(output / "export"),
                "--tokenizer-dir",
                str(root / "tokenizer"),
            ],
        ),
        (
            "development",
            [
                sys.executable,
                "-m",
                "scripts.capability_eval",
                *evaluation,
                "--local-export",
                str(output / "export"),
                "--partition",
                "development",
                "--limit",
                "0",
                "--output",
                str(output / "development"),
            ],
        ),
    ]


def launch_environment(root):
    return {
        **ENVIRONMENT,
        "HF_HUB_CACHE": str(Path(root) / "hub"),
        "NLTK_DATA": str(Path(root) / "nltk"),
    }


def write_launch(root, output, manifest):
    bind(root, output)
    phases = commands(root, output)
    environment = launch_environment(root)
    durable_json(
        Path(output) / "launch.json",
        {"commands": phases, "environment": environment, "commit": manifest["commit"]},
    )
    return phases, environment


def preflight(root, output):
    import importlib.metadata
    import platform

    import nltk
    import torch

    from speck.evaluation.code_runner import check_sandbox

    check_sandbox()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("exactly one visible allocated GPU is required")
    device = torch.cuda.get_device_properties(0)
    if "H100" not in device.name or platform.machine() != "x86_64":
        raise ValueError("this launch packet is qualified for x86_64 H100 only")
    if device.total_memory < 75 * 1024**3 or torch.version.cuda != "12.8":
        raise ValueError("expected a full H100 with the qualified CUDA 12.8 runtime")
    if sys.version_info[:2] != (3, 10):
        raise ValueError("use the qualified Python 3.10 runtime")
    if shutil.disk_usage(output).free < 128 * 1024**3:
        raise ValueError("at least 128 GiB free checkpoint space is required")
    if torch.cuda.mem_get_info()[0] < 24 * 1024**3:
        raise ValueError("at least 24 GiB free GPU memory is required")
    nltk.data.find("tokenizers/punkt_tab/english/")
    packages = {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "triton",
            "flash-linear-attention",
            "liger-kernel",
            "transformers",
            "lm_eval",
            "evalplus",
        )
    }
    # Runtime versions used by the passing H100 rehearsal. Other stacks need qualification.
    for name, expected in {
        "torch": "2.9.1",
        "triton": "3.5.1",
        "flash-linear-attention": "0.5.0",
        "liger-kernel": "0.8.2",
    }.items():
        if packages[name].split("+")[0] != expected:
            raise ValueError(f"unqualified runtime version: {name}={packages[name]}")
    durable_json(
        Path(output) / "environment.json",
        {
            "gpu": device.name,
            "vram_bytes": device.total_memory,
            "machine": platform.machine(),
            "python": sys.version,
            "cuda": torch.version.cuda,
            "packages": packages,
            "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True),
        },
    )


def check_ledger(ledger, prior, directory):
    if (
        type(prior) not in (int, float)
        or not math.isfinite(prior)
        or not 0 <= prior <= CEILING_HOURS
    ):
        raise ValueError("prior GPU-hours must be a finite nonnegative accounted amount")
    if (
        ledger.get("format") != FORMAT
        or type(ledger.get("prior_gpu_hours")) not in (int, float)
        or not math.isfinite(ledger["prior_gpu_hours"])
        or not 0 <= ledger["prior_gpu_hours"] <= prior
        or ledger.get("ceiling_gpu_hours") != CEILING_HOURS
    ):
        raise ValueError("ledger identity/accounting changed; reconcile before execution")
    spent = prior
    for row in ledger["attempts"]:
        if row["state"] == "reserved_or_running":
            raise ValueError("unresolved attempt: inspect processes and reconcile before execution")
        if row["state"] not in ("completed", "failed"):
            raise ValueError("invalid ledger attempt state")
        for name in ("reserved_gpu_hours", "observed_gpu_hours"):
            value = row[name]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid attempt accounting")
        result = Path(directory) / row["id"] / "result.json"
        if (
            not result.resolve().is_relative_to(Path(directory).resolve())
            or file_sha256(result) != row["result_sha256"]
        ):
            raise ValueError("prior attempt result changed")
        report = json.loads(result.read_text())
        if (
            row["reserved_gpu_hours"] != SESSION_SECONDS / 3600
            or row["observed_gpu_hours"] != report["observed_gpu_hours"]
            or row["state"] != report["status"]
        ):
            raise ValueError("attempt accounting differs from retained result")
        spent += max(row["reserved_gpu_hours"], row["observed_gpu_hours"])
    if spent + SESSION_SECONDS / 3600 > CEILING_HOURS:
        raise ValueError("six-hour reservation exceeds remaining pilot budget")
    return spent


def execute_phases(phases, output, deadline):
    results = {}
    for name, command in phases:
        remaining = deadline - time.monotonic() - GRACE_SECONDS - 5
        if remaining <= 0:
            raise TimeoutError("shared pilot deadline exhausted")
        phase_dir = Path(output) / "logs" / name
        phase_dir.mkdir(parents=True)
        execution = supervise(command, phase_dir, remaining, GRACE_SECONDS)
        results[name] = execution
        durable_json(Path(output) / "phases.json", results)
        if execution["termination"] != "exited" or execution["returncode"] != 0:
            raise RuntimeError(f"pilot phase failed: {name}: {execution['termination']}")
    return results


def run(root, ledger_dir, prior):
    root, ledger_dir = Path(root).resolve(), Path(ledger_dir).resolve()
    manifest = verify_packet(root)
    ledger_dir.mkdir(parents=True, exist_ok=True)
    with (ledger_dir / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger_path = ledger_dir / "ledger.json"
        ledger = (
            json.loads(ledger_path.read_text())
            if ledger_path.exists()
            else {
                "format": FORMAT,
                "prior_gpu_hours": prior,
                "ceiling_gpu_hours": CEILING_HOURS,
                "attempts": [],
            }
        )
        check_ledger(ledger, prior, ledger_dir)
        ledger["prior_gpu_hours"] = prior
        attempt_id = uuid.uuid4().hex
        output = ledger_dir / attempt_id
        output.mkdir()
        row = {
            "id": attempt_id,
            "state": "reserved_or_running",
            "reserved_gpu_hours": SESSION_SECONDS / 3600,
            "packet_sha256": manifest["sha256"],
        }
        ledger["attempts"].append(row)
        durable_json(ledger_path, ledger)
        started = time.monotonic()
        result = {
            "status": "running",
            "allocated_gpus": 1,
            "packet_sha256": manifest["sha256"],
            "session_seconds": SESSION_SECONDS,
        }
        durable_json(output / "result.json", result)
        previous_environment = dict(os.environ)
        try:
            phases, environment = write_launch(root, output, manifest)
            os.environ.update(environment)
            result["phases"] = execute_phases(phases, output, started + SESSION_SECONDS)
            summary = json.loads((output / "checkpoints/run_summary.json").read_text())
            if (
                summary.get("status") != "completed"
                or summary.get("completed_steps") != 800
                or summary.get("global_tokens") != 104857600
                or summary.get("partial") is not False
            ):
                raise ValueError("pilot endpoint did not match the frozen experiment")
            result["status"] = "completed"
        except BaseException as error:
            result.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            os.environ.clear()
            os.environ.update(previous_environment)
            result["observed_gpu_hours"] = (time.monotonic() - started) / 3600
            result["boundary"] = (
                "One allocated GPU. Provider idle/setup/transfer and earlier untracked usage must be accounted separately; this supervisor never stops provider billing. No automatic retry or resume."
            )
            durable_json(output / "result.json", result)
            row.update(
                state=result["status"],
                observed_gpu_hours=result["observed_gpu_hours"],
                result_sha256=file_sha256(output / "result.json"),
            )
            durable_json(ledger_path, ledger)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    pack = sub.add_parser("prepare")
    for name in ("output", "data", "tokenizer", "evaluation", "nltk-data"):
        pack.add_argument("--" + name, required=True, type=Path)
    for action in ("check", "run", "preflight"):
        command = sub.add_parser(action)
        command.add_argument("root", type=Path)
        if action == "check":
            command.add_argument(
                "--output", type=Path, help="write a CPU-only launch preview to a fresh directory"
            )
        if action == "run":
            command.add_argument("--ledger", required=True, type=Path)
            command.add_argument("--prior-gpu-hours", required=True, type=float)
        if action == "preflight":
            command.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        result = prepare(args.output, args.data, args.tokenizer, args.evaluation, args.nltk_data)
    elif args.action == "check":
        result = verify_packet(args.root)
        if args.output is not None:
            args.output.mkdir(parents=True, exist_ok=False)
            write_launch(args.root.resolve(), args.output.resolve(), result)
    elif args.action == "preflight":
        result = preflight(args.root, args.output)
    else:
        result = run(args.root, args.ledger, args.prior_gpu_hours)
    print(json.dumps(result, indent=2))
