import argparse
import json

import pytest

from scripts.inference_benchmark import _batch_sizes, _percentile, _speck_identity


def test_batch_sizes():
    assert _batch_sizes("1,32") == (1, 32)
    with pytest.raises(argparse.ArgumentTypeError):
        _batch_sizes("1,0")


def test_percentile_uses_nearest_rank():
    assert _percentile([4.0, 1.0, 3.0, 2.0], 0.25) == 2.0
    assert _percentile([4.0, 1.0, 3.0, 2.0], 0.75) == 3.0


def test_speck_identity_uses_selected_experiment_name_and_records_source(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints" / "canonical-run"
    checkpoint_dir.mkdir(parents=True)
    model_path = checkpoint_dir / "model_000123.pt"
    metadata_path = checkpoint_dir / "metadata_000123.json"
    model_path.write_bytes(b"model")
    metadata = {
        "step": 123,
        "resolved": {"run": "historical-run", "experiment": "/old/experiment"},
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    name, checkpoint = _speck_identity(
        tmp_path / "experiments" / "canonical-run",
        "canonical-run",
        checkpoint_dir,
        123,
        metadata,
    )

    assert name == "canonical-run"
    assert checkpoint["source_run"] == "historical-run"
    assert checkpoint["source_experiment"] == "/old/experiment"
    assert checkpoint["model_sha256"] != checkpoint["metadata_sha256"]
