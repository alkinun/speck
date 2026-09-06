"""Qualify systemd read-only path isolation on disposable systems-workload fixtures."""

import json
import os
import platform
import subprocess
import tempfile
import uuid
from pathlib import Path

from speck.paper_finalist_systems_workload import file_sha256, require_output_outside_protected

CHILD = """\
import errno
import json
import sys
from pathlib import Path

protected = Path(sys.argv[1])
receipt = Path(sys.argv[2])
target = protected / "forbidden-write.txt"
try:
    target.write_text("mutation\\n", encoding="utf-8")
except OSError as error:
    receipt.write_text(json.dumps({"write_blocked": True, "errno": error.errno}) + "\\n", encoding="utf-8")
    if error.errno not in {errno.EROFS, errno.EACCES, errno.EPERM}:
        raise
else:
    receipt.write_text(json.dumps({"write_blocked": False, "errno": None}) + "\\n", encoding="utf-8")
    raise SystemExit(7)
"""


def build_systemd_command(unit, protected_path, writable_path, receipt_path):
    if not isinstance(unit, str) or not unit.startswith("speck-paper1-systems-sandbox-"):
        raise ValueError("systems sandbox unit name is invalid")
    protected_path = Path(protected_path).resolve()
    writable_path = require_output_outside_protected(writable_path, [protected_path])
    receipt_path = Path(receipt_path).resolve()
    if not receipt_path.parent == writable_path or receipt_path.name != "receipt.json":
        raise ValueError("systems sandbox receipt must use the exact writable fixture directory")
    return [
        "systemd-run",
        "--user",
        "--wait",
        "--collect",
        "--pipe",
        "--quiet",
        f"--unit={unit}",
        "--property=Type=exec",
        "--property=Restart=no",
        f"--property=ReadOnlyPaths={protected_path}",
        f"--property=ReadWritePaths={writable_path}",
        "/usr/bin/python3",
        "-c",
        CHILD,
        str(protected_path),
        str(receipt_path),
    ]


def _run(command):
    return subprocess.run(command, check=False, capture_output=True, text=True)


def qualify_disposable_sandbox(*, runner=_run, temporary_root=None, unit_suffix=None):
    manager = None
    if temporary_root is None:
        manager = tempfile.TemporaryDirectory(prefix="speck-systems-sandbox-")
        root = Path(manager.name)
    else:
        root = Path(temporary_root).resolve()
        root.mkdir(parents=True, exist_ok=False)
    try:
        protected = root / "protected"
        writable = root / "writable"
        protected.mkdir()
        writable.mkdir()
        sentinel = protected / "sentinel.txt"
        sentinel.write_text("immutable sentinel\n", encoding="utf-8")
        sentinel_sha = file_sha256(sentinel)
        receipt = writable / "receipt.json"
        suffix = unit_suffix or f"{os.getpid()}-{uuid.uuid4().hex[:12]}"
        unit = f"speck-paper1-systems-sandbox-{suffix}"
        version = runner(["systemd-run", "--version"])
        if version.returncode != 0:
            raise RuntimeError("cannot identify systemd-run for systems sandbox")
        command = build_systemd_command(unit, protected, writable, receipt)
        completed = runner(command)
        if completed.returncode != 0:
            raise RuntimeError(
                "disposable systems sandbox service failed: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        if not receipt.is_file():
            raise RuntimeError("disposable systems sandbox produced no receipt")
        value = json.loads(receipt.read_text(encoding="utf-8"))
        if value.get("write_blocked") is not True or value.get("errno") not in {1, 13, 30}:
            raise RuntimeError("disposable systems sandbox did not block the protected write")
        if file_sha256(sentinel) != sentinel_sha or (protected / "forbidden-write.txt").exists():
            raise RuntimeError("disposable systems sandbox changed the protected fixture")
        return {
            "status": "qualified",
            "unit": unit,
            "systemd": version.stdout.splitlines()[0].strip(),
            "kernel": platform.release(),
            "read_only_paths": str(protected),
            "read_write_paths": str(writable),
            "write_blocked": True,
            "errno": value["errno"],
            "sentinel_sha256_before": sentinel_sha,
            "sentinel_sha256_after": file_sha256(sentinel),
            "forbidden_path_absent": True,
            "wait": True,
            "collect": True,
            "polling": False,
            "finalist_checkpoint_accessed": False,
        }
    finally:
        if manager is not None:
            manager.cleanup()
