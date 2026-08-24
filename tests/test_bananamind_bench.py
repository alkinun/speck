import hashlib
import json
from types import SimpleNamespace

import torch

from scripts import bananamind_bench


class FakeTokenizer:
    vocab_size = 32
    bos_id = 1
    eos_id = 2

    def fingerprint(self):
        return "tokenizer-fingerprint"


def test_speck_checkpoint_option_is_removed_from_official_arguments():
    step, remaining = bananamind_bench._parse_speck_options(
        ["--model", "experiment", "--speck-checkpoint-step", "42", "--resume"]
    )

    assert step == 42
    assert remaining == ["--model", "experiment", "--resume"]


def test_official_dataset_defaults_to_pinned_revision():
    args = SimpleNamespace(dataset_id=bananamind_bench.DATASET_ID, dataset_revision="main")

    bananamind_bench._pin_dataset_revision(args)

    assert args.dataset_revision == bananamind_bench.DATASET_REVISION


def test_resolve_speck_run_pins_checkpoint_and_tokenizer(tmp_path, monkeypatch):
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    tokenizer_config = {
        "directory": None,
        "filename": "tokenizer.model",
        "repo": "example/tokenizer",
        "revision": "tokenizer-revision",
    }
    (experiment / "model.json").write_text("{}")
    (experiment / "tokenizer.json").write_text(json.dumps(tokenizer_config))
    (experiment / "train.json").write_text(json.dumps({"output_dir": None, "run": "test"}))

    checkpoint_dir = tmp_path / "checkpoints" / "test"
    checkpoint_dir.mkdir(parents=True)
    model_bytes = b"model checkpoint"
    (checkpoint_dir / "model_000042.pt").write_bytes(model_bytes)
    (checkpoint_dir / "complete_000042").write_text("complete\n")
    metadata = {
        "step": 42,
        "config": {"vocab_size": 32, "bos_token_id": 1, "eos_token_id": 2},
        "resolved": {"tokenizer": tokenizer_config},
    }
    metadata_path = checkpoint_dir / "metadata_000042.json"
    metadata_path.write_text(json.dumps(metadata))

    monkeypatch.setattr(bananamind_bench, "base_dir", lambda: str(tmp_path))
    monkeypatch.setattr(bananamind_bench, "get_tokenizer", lambda **config: FakeTokenizer())
    args = SimpleNamespace(model=str(experiment), tokenizer=None)

    run = bananamind_bench._resolve_speck_run(args, 42)

    assert run["step"] == 42
    assert run["identity"]["checkpoint_sha256"] == hashlib.sha256(model_bytes).hexdigest()
    assert run["identity"]["tokenizer_sha256"] == "tokenizer-fingerprint"


def test_run_identity_captures_numerical_configuration():
    args = SimpleNamespace(
        batch_size=32,
        device="auto",
        dtype="auto",
        threads=8,
        _speck_run={"identity": {"checkpoint_step": 42}},
    )
    official = SimpleNamespace(
        resolve_device=lambda value: torch.device("cpu"),
        resolve_dtype=lambda value, device: torch.float32,
    )

    signature = bananamind_bench._add_run_identity({"model": "Speck"}, args, official)

    assert signature["batch_size"] == 32
    assert signature["device"] == "cpu"
    assert signature["dtype"] == "float32"
    assert signature["speck"] == {"checkpoint_step": 42}
