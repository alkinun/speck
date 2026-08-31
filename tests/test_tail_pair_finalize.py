import json

import pytest
import torch

from scripts.tail_pair_finalize import finalize
from speck.checkpoint import checkpoint_identity, save


def save_arm(directory, arm, run, parent, values, global_offset=100):
    for step, value in enumerate(values, 1):
        metadata = {
            "step": step,
            "global_tokens": global_offset + step * 10,
            "global_step": global_offset // 10 + step,
            "training_phase": "base",
            "config": {"model": "same"},
            "manifest": "data",
            "data_state": {"global_consumed_tokens": global_offset + step * 10},
            "resolved": {
                "run": run,
                "steps": len(values),
                "train_tokens": len(values) * 10,
                "consumed_tokens": len(values) * 10,
                "world_size": 1,
                "global_token_offset": global_offset,
                "parent_checkpoint": parent,
                "branch_schedule": "inherit" if arm == "control" else "new",
            },
        }
        save(
            directory,
            step,
            {"weight": torch.tensor([value], dtype=torch.float32)},
            {"step": step},
            metadata,
        )


def test_finalize_builds_matched_average_artifacts(tmp_path):
    parent_dir = tmp_path / "parent"
    save(parent_dir, 10, {}, {}, {"step": 10})
    parent = checkpoint_identity(parent_dir, 10)
    pair_dir = tmp_path / "pair"
    pair_dir.mkdir()
    pair = {
        "format": "speck_tail_pair",
        "format_version": 1,
        "parent_checkpoint": parent,
        "parent_global_tokens": 100,
        "manifest": "data",
        "train_tokens": 30,
        "consumed_tokens": 30,
        "world_size": 1,
        "control": {"run": "pair-Control", "schedule": "inherit"},
        "constant": {"run": "pair-Constant", "schedule": "new"},
    }
    (pair_dir / "pair.json").write_text(json.dumps(pair))
    control = tmp_path / "control"
    constant = tmp_path / "constant"
    save_arm(control, "control", "pair-Control", parent, [1.0, 2.0, 3.0])
    save_arm(constant, "constant", "pair-Constant", parent, [2.0, 4.0, 6.0])

    output = tmp_path / "final"
    report = finalize(pair_dir, control, constant, [1, 2, 3], output)

    assert report["global_tokens"] == 130
    assert report["average_steps"] == [1, 2, 3]
    control_state = torch.load(output / "control-average" / "model.pt")
    constant_state = torch.load(output / "constant-average" / "model.pt")
    assert control_state["weight"].item() == 2.0
    assert constant_state["weight"].item() == 4.0
    assert (output / "finalization.json").is_file()


def test_finalize_rejects_mismatched_parent_or_window(tmp_path):
    parent_dir = tmp_path / "parent"
    save(parent_dir, 10, {}, {}, {"step": 10})
    parent = checkpoint_identity(parent_dir, 10)
    pair_dir = tmp_path / "pair"
    pair_dir.mkdir()
    pair = {
        "format": "speck_tail_pair",
        "format_version": 1,
        "parent_checkpoint": parent,
        "parent_global_tokens": 100,
        "manifest": "data",
        "train_tokens": 20,
        "consumed_tokens": 20,
        "world_size": 1,
        "control": {"run": "control", "schedule": "inherit"},
        "constant": {"run": "constant", "schedule": "new"},
    }
    (pair_dir / "pair.json").write_text(json.dumps(pair))
    control = tmp_path / "control"
    constant = tmp_path / "constant"
    save_arm(control, "control", "control", parent, [1.0, 2.0])
    save_arm(constant, "constant", "constant", {**parent, "step": 9}, [1.0, 2.0])

    with pytest.raises(ValueError, match="does not match"):
        finalize(pair_dir, control, constant, [1, 2], tmp_path / "bad-parent")
    with pytest.raises(ValueError, match="sorted unique"):
        finalize(pair_dir, control, control, [2, 1], tmp_path / "bad-window")
