import json

import pytest
import torch

from scripts.checkpoint_average import (
    average_checkpoints,
    average_identity,
    parse_args,
    write_average,
)
from speck.checkpoint import save


def metadata(step, tokens, manifest="data"):
    return {
        "step": step,
        "global_tokens": tokens,
        "training_phase": "base",
        "config": {"model": "same"},
        "manifest": manifest,
        "resolved": {"run": "run", "steps": 10},
    }


def checkpoint(directory, step, value, *, manifest="data", counter=1):
    save(
        directory,
        step,
        {
            "weight": torch.tensor([value], dtype=torch.bfloat16),
            "counter": torch.tensor(counter),
        },
        {"step": step},
        metadata(step, step * 10, manifest),
    )


def test_average_checkpoints_writes_model_only_artifact(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    checkpoint(checkpoints, 1, 1.0)
    checkpoint(checkpoints, 2, 2.0)
    checkpoint(checkpoints, 3, 6.0)

    state, lineage = average_checkpoints(checkpoints, [1, 2, 3])
    output = tmp_path / "average"
    write_average(output, state, lineage)

    stored = torch.load(output / "model.pt", map_location="cpu")
    stored_lineage = json.loads((output / "metadata.json").read_text())
    assert stored["weight"].dtype == torch.bfloat16
    assert stored["weight"].item() == pytest.approx(3.0)
    assert stored["counter"].item() == 1
    assert stored_lineage["format"] == "speck_model_average"
    assert stored_lineage["average"] == {
        "accumulation_dtype": "float32",
        "count": 3,
        "weight": 1 / 3,
    }
    assert [item["step"] for item in stored_lineage["checkpoints"]] == [1, 2, 3]
    assert all(len(item["optimizer_sha256"]) == 64 for item in stored_lineage["checkpoints"])
    assert (output / "complete").read_text() == "complete\n"
    identity = average_identity(output)
    assert identity["directory"] == str(output.resolve())
    assert len(identity["model_sha256"]) == len(identity["metadata_sha256"]) == 64


def test_average_rejects_incompatible_checkpoints(tmp_path):
    checkpoint(tmp_path, 1, 1.0)
    checkpoint(tmp_path, 2, 2.0, manifest="other")
    with pytest.raises(ValueError, match="trajectory"):
        average_checkpoints(tmp_path, [1, 2])

    checkpoint(tmp_path, 2, 2.0, counter=2)
    with pytest.raises(ValueError, match="non-floating"):
        average_checkpoints(tmp_path, [1, 2])

    torch.save(
        {"weight": torch.ones(2, dtype=torch.bfloat16), "counter": torch.tensor(1)},
        tmp_path / "model_000002.pt",
    )
    with pytest.raises(ValueError, match="tensor does not match"):
        average_checkpoints(tmp_path, [1, 2])


def test_average_arguments_require_explicit_steps_and_output():
    args = parse_args(["checkpoints", "--steps", "2", "4", "--output-dir", "average"])
    assert args.steps == [2, 4]
    assert str(args.output_dir) == "average"
