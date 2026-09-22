import json
import subprocess
import sys
from pathlib import Path

import pytest

from speck.operations.gh200 import bind, bundle, export_arguments


@pytest.mark.parametrize("phase", ["base", "sft"])
def test_rental_export_uses_matching_cli_and_never_uploads(phase, tmp_path, monkeypatch):
    from speck.export import checkpoint, transformers

    command = export_arguments(
        phase, tmp_path / "checkpoint", tmp_path / "export", tmp_path / "tokenizer"
    )
    monkeypatch.setattr(sys, "argv", command[1:])
    if phase == "base":
        assert command[1] == "scripts.base_checkpoint_export"
        args = checkpoint.arguments()
        assert args.tokenizer_dir == tmp_path / "tokenizer"
    else:
        assert command[1] == "scripts.model_publish"
        args = transformers.arguments()
        assert args.no_upload is True
        assert args.expected_epochs == 1
    assert args.checkpoint_dir == Path(tmp_path / "checkpoint")
    assert args.output_dir == tmp_path / "export"
    assert args.step == 4


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
        (path / "documents.jsonl").write_text('{"ordinal":0}\n')
        inputs.append(path)
    output = tmp_path / "rental"
    result = bundle(output, *inputs)
    assert (output / "pilot-data/documents.jsonl").read_text() == '{"ordinal":0}\n'
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
    # Execute the packet's experiment lookup from the transported checkout, where the original
    # workstation tokenizer path is unavailable. Both hardware packets use the same bundle layout.
    from speck.config import load_experiment

    for name in ("h100", "gh200"):
        packet = (
            Path(__file__).resolve().parents[2]
            / f"experiments/qualification/throughput-{name}.json"
        )
        common = json.loads(packet.read_text())["common"]
        assert load_experiment(common["experiment"], *configs) == configs
        assert Path(common["data_dir"]).resolve() == output / "pilot-data"
        assert not Path(common["output_directory"]).resolve().is_relative_to(output / "code")
    (output / "pilot-data/payload.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity mismatch"):
        bind(output)
