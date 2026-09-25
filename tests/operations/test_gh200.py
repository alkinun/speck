import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from speck.operations.gh200 import bind, bundle, export_arguments


@pytest.mark.parametrize("phase", ["base", "sft"])
def test_export_uses_matching_cli_and_never_uploads(phase, tmp_path, monkeypatch):
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
    # The ladder configurations the bind step relocates, copied from this checkout.
    source = Path(__file__).resolve().parents[2]
    ladder = ["experiments/main-data/plan.json", "experiments/ladder/train.json"]
    ladder += [f"experiments/ladder/{rung}/model.json" for rung in ("50m", "130m", "410m")]
    for name in ladder:
        (repository / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, repository / name)
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
    output = tmp_path / "bundle"
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
    # Execute the packet's experiment lookups from the transported checkout, where the original
    # tokenizer path is unavailable.
    from speck.config import load_experiment

    packet = json.loads(
        (
            Path(__file__).resolve().parents[2] / "experiments/qualification/throughput-gh200.json"
        ).read_text()
    )
    common = packet["common"]
    assert load_experiment(common["experiment"], *configs) == configs
    for run in packet["runs"]:
        experiment = run.get("overrides", {}).get("experiment", common["experiment"])
        loaded = load_experiment(experiment, "tokenizer", "model", "train")
        assert loaded["tokenizer"] == configs["tokenizer"]
    rung = load_experiment("../relocated-410m", "model", "train")
    assert rung["model"] == load_experiment("experiments/ladder/410m", "model")["model"]
    assert rung["train"]["batch_tokens"] == 131072
    assert Path(common["data_dir"]).resolve() == output / "pilot-data"
    assert not Path(common["output_directory"]).resolve().is_relative_to(output / "code")
    (output / "pilot-data/payload.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity mismatch"):
        bind(output)
