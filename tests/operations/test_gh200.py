import json
import subprocess

import pytest

from speck.operations.gh200 import bind, bundle


def test_portable_bundle_relocates_clean_checkout_and_detects_tampering(tmp_path, monkeypatch):
    repository = tmp_path / "source"
    repository.mkdir()
    monkeypatch.chdir(repository)
    subprocess.run(["git", "init", "-b", "codex/fixture"], check=True, capture_output=True)
    experiment = repository / "experiments/pilot"
    experiment.mkdir(parents=True)
    for name, value in {
        "model": {},
        "train": {},
        "data": {"output_name": "local"},
        "tokenizer": {"directory": "/old/machine/tokenizer"},
    }.items():
        (experiment / f"{name}.json").write_text(json.dumps(value))
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-m",
            "fixture",
        ],
        check=True,
        capture_output=True,
    )
    inputs = []
    for name in ("data", "tokenizer", "assistant"):
        path = tmp_path / name
        path.mkdir()
        (path / "payload.bin").write_bytes(b"frozen")
        inputs.append(path)
    output = tmp_path / "rental"
    result = bundle(output, *inputs)
    subprocess.run(
        [
            "git",
            "clone",
            "--branch",
            result["branch"],
            str(output / "code.bundle"),
            str(output / "code"),
        ],
        check=True,
        capture_output=True,
    )
    monkeypatch.chdir(output / "code")
    _, configs, directory = bind(output)
    assert directory.is_dir()
    assert configs["tokenizer"]["directory"] == str(output / "tokenizer")
    assert configs["data"]["output_dir"] == str(output / "pilot-data")
    assert "output_name" not in configs["data"]
    (output / "pilot-data/payload.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity mismatch"):
        bind(output)
