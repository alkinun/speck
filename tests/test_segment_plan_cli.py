import hashlib
import json
from types import SimpleNamespace

import speck.search.segments as segments
from scripts.segment_plan import create


def record(name, split, start, end):
    return segments.DocumentRecord(
        hashlib.sha256(name.encode()).hexdigest(),
        split,
        start,
        end,
        "test",
        1.0,
    )


def test_segment_plan_command_writes_a_hashed_plan(tmp_path, monkeypatch):
    records = tuple(
        [record(f"train-{index}", "train", index * 10, index * 10 + 10) for index in range(6)]
        + [record(f"val-{index}", "val", index * 10, index * 10 + 10) for index in range(6)]
    )

    def build(data_dir, data_seed, train_tokens, validation_tokens):
        return segments.build_segment_plan(
            records,
            "dataset",
            data_seed,
            train_tokens,
            validation_tokens,
        )

    monkeypatch.setattr("scripts.segment_plan.build_segment_plan_from_dataset", build)
    output = tmp_path / "segments.json"
    result = create(
        SimpleNamespace(
            data_dir=tmp_path,
            output=output,
            data_seed=42,
            train_tokens=20,
            monitor_tokens=10,
            promotion_tokens=10,
            audit_tokens=10,
            final_tokens=10,
        )
    )
    stored = segments.SegmentPlan.from_dict(json.loads(output.read_text()))
    assert stored.digest == result["digest"]
    assert result["partitions"] == {
        "audit": 10,
        "final": 10,
        "monitor": 10,
        "promotion": 10,
        "train": 20,
    }
