"""Run a materialized MoE screen serially with resumable checkpoints and volume-local logs."""

import argparse
import getpass
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from scripts.moe_screen_analyze import file_sha256
from speck.checkpoint import latest


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("screen", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--minimum-free-gib", type=float, default=100.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.minimum_free_gib) or args.minimum_free_gib < 1:
        parser.error("--minimum-free-gib must be a finite value of at least one")
    return args


def load_and_verify_screen(screen):
    screen = Path(screen).resolve()
    contract_path = screen / "screen.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("format") != "speck_moe_design_screen" or contract.get("format_version") != 2:
        raise ValueError("runner requires a version-2 MoE design screen")
    if len(contract["launch_order"]) != len(set(contract["launch_order"])) or set(
        contract["launch_order"]
    ) != set(contract["arms"]):
        raise ValueError("screen launch order does not cover every arm exactly once")
    for name, arm in contract["arms"].items():
        for filename, expected in arm["artifacts"].items():
            path = screen / name / filename
            if not path.is_file() or file_sha256(path) != expected:
                raise ValueError(f"screen artifact changed: {name}/{filename}")
    repository = Path(__file__).parents[1]
    for relative, expected in contract.get("implementation_artifacts", {}).items():
        path = repository / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"screen implementation changed: {relative}")
    return screen, contract


def command_for_arm(screen, output_root, arm, resume=None):
    command = [
        sys.executable,
        "-m",
        "scripts.base_train",
        str(screen / arm),
        "--output-dir",
        str(output_root / f"SpeckLC-150M-MoEScreen-{arm}"),
    ]
    if resume is not None:
        command.extend(("--resume", str(resume)))
    return command


def append_event(path, **values):
    record = {"at": datetime.now(timezone.utc).isoformat(), **values}
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def runtime_environment(output_root):
    environment = dict(os.environ)
    environment["WANDB_DIR"] = str(Path(output_root) / "wandb")
    # Inductor imports generated shared libraries from its cache. The checkpoint
    # volume may be mounted noexec, so compiled artifacts must remain on an
    # executable filesystem rather than beside logs and weights.
    environment["TORCHINDUCTOR_CACHE_DIR"] = str(
        Path(tempfile.gettempdir()) / f"torchinductor_{getpass.getuser()}"
    )
    return environment


def run(args):
    screen, contract = load_and_verify_screen(args.screen)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    logs = output_root / "moe-screen-logs"
    logs.mkdir(exist_ok=True)
    events = logs / "events.jsonl"
    environment = runtime_environment(output_root)
    Path(environment["WANDB_DIR"]).mkdir(exist_ok=True)
    Path(environment["TORCHINDUCTOR_CACHE_DIR"]).mkdir(exist_ok=True)
    minimum_free = int(args.minimum_free_gib * 2**30)
    expected_steps = math.ceil(contract["train_tokens"] / 65_536)

    commands = []
    for arm in contract["launch_order"]:
        directory = output_root / f"SpeckLC-150M-MoEScreen-{arm}"
        resume = latest(directory)
        if resume is not None and resume >= expected_steps:
            append_event(events, arm=arm, status="skipped_complete", step=resume)
            continue
        free = shutil.disk_usage(output_root).free
        if free < minimum_free:
            raise RuntimeError(
                f"refusing to start {arm}: {free / 2**30:.1f} GiB free is below "
                f"the {args.minimum_free_gib:.1f} GiB floor"
            )
        command = command_for_arm(screen, output_root, arm, resume)
        commands.append(command)
        if args.dry_run:
            continue
        append_event(events, arm=arm, status="started", resume=resume, command=command)
        log_path = logs / f"{arm}.log"
        with log_path.open("a" if resume is not None else "w", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                cwd=Path(__file__).parents[1],
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode:
            append_event(events, arm=arm, status="failed", returncode=completed.returncode)
            raise RuntimeError(
                f"{arm} failed with exit code {completed.returncode}; see {log_path}"
            )
        append_event(events, arm=arm, status="completed")
    return commands


def main(argv=None):
    args = arguments(argv)
    commands = run(args)
    if args.dry_run:
        for command in commands:
            print(" ".join(command))


if __name__ == "__main__":
    main()
