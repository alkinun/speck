import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from speck.paper_finalist_systems_sandbox import (
    build_systemd_command,
    qualify_disposable_sandbox,
)


def fake_runner(command):
    if command == ["systemd-run", "--version"]:
        return SimpleNamespace(returncode=0, stdout="systemd 258 (258.1)\n", stderr="")
    protected = Path(command[-2])
    receipt = Path(command[-1])
    assert protected.name == "protected"
    receipt.write_text(json.dumps({"write_blocked": True, "errno": 30}) + "\n", encoding="utf-8")
    return SimpleNamespace(returncode=0, stdout="", stderr="")


def test_systemd_sandbox_command_is_waited_collected_and_path_scoped(tmp_path):
    protected = tmp_path / "protected"
    writable = tmp_path / "writable"
    protected.mkdir()
    writable.mkdir()
    command = build_systemd_command(
        "speck-paper1-systems-sandbox-test",
        protected,
        writable,
        writable / "receipt.json",
    )
    assert command[:6] == ["systemd-run", "--user", "--wait", "--collect", "--pipe", "--quiet"]
    assert f"--property=ReadOnlyPaths={protected}" in command
    assert f"--property=ReadWritePaths={writable}" in command
    assert "--property=Restart=no" in command


def test_systemd_sandbox_command_rejects_path_overlap(tmp_path):
    protected = tmp_path / "protected"
    protected.mkdir()
    with pytest.raises(ValueError, match="overlaps"):
        build_systemd_command(
            "speck-paper1-systems-sandbox-test",
            protected,
            protected / "writable",
            protected / "writable" / "receipt.json",
        )


def test_mocked_disposable_sandbox_qualifies_and_cleans(tmp_path):
    root = tmp_path / "fixture"
    report = qualify_disposable_sandbox(
        runner=fake_runner,
        temporary_root=root,
        unit_suffix="test",
    )
    assert report["status"] == "qualified"
    assert report["write_blocked"] is True
    assert report["sentinel_sha256_before"] == report["sentinel_sha256_after"]
    assert report["wait"] is True
    assert report["collect"] is True
    assert report["polling"] is False


def test_mocked_disposable_sandbox_rejects_successful_protected_write(tmp_path):
    def unsafe_runner(command):
        if command == ["systemd-run", "--version"]:
            return SimpleNamespace(returncode=0, stdout="systemd 258\n", stderr="")
        receipt = Path(command[-1])
        receipt.write_text(
            json.dumps({"write_blocked": False, "errno": None}) + "\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with pytest.raises(RuntimeError, match="did not block"):
        qualify_disposable_sandbox(
            runner=unsafe_runner,
            temporary_root=tmp_path / "unsafe",
            unit_suffix="unsafe",
        )


def test_mocked_disposable_sandbox_rejects_service_failure(tmp_path):
    def failed_runner(command):
        if command == ["systemd-run", "--version"]:
            return SimpleNamespace(returncode=0, stdout="systemd 258\n", stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="namespace failure")

    with pytest.raises(RuntimeError, match="namespace failure"):
        qualify_disposable_sandbox(
            runner=failed_runner,
            temporary_root=tmp_path / "failed",
            unit_suffix="failed",
        )
