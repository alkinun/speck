"""Translated benchmark tasks must not cancel each other's exclusion anchors."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "check_qualification",
    Path(__file__).resolve().parents[2] / "experiments/main-data/check_qualification.py",
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_shared_prompt_across_lanes_still_matches(tmp_path):
    text = "Return the distinct positive integer values sorted into descending numerical order."
    benchmarks = []
    for lang in ("python", "javascript"):
        path = tmp_path / (lang + ".jsonl")
        path.write_text(json.dumps({"id": "one", "prompt": text}) + "\n")
        benchmarks.append(
            {
                "id": lang,
                "path": str(path),
                "format": "jsonl",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "expected_tasks": 1,
                "task_id_field": "id",
                "text_fields": ["prompt"],
            }
        )
    policy = json.loads(
        (Path(__file__).resolve().parents[2] / "experiments/pilot/evaluation.json").read_text()
    )["exclusion_policy"]
    result = audit.screen([{"id": "candidate", "text": text}], benchmarks, policy)
    assert result == {"candidate": {"python": 1, "javascript": 1}}


def test_changed_artifact_is_rejected(tmp_path):
    path = tmp_path / "payload"
    path.write_text("changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        audit.verified({"path": str(path), "sha256": "0" * 64})


def test_changed_split_policy_is_not_silently_ignored(tmp_path):
    records = tmp_path / "records.json"
    records.write_text("[]")
    rules = tmp_path / "rules.json"
    rules.write_text(
        json.dumps(
            {
                "family_split": {
                    "train_buckets": 8000,
                    "development_buckets": 1000,
                    "final_buckets": 1000,
                    "total_buckets": 10000,
                }
            }
        )
    )

    def artifact(path):
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    with pytest.raises(ValueError, match="unsupported family split"):
        audit.audit({"evidence": [], "records": artifact(records), "rules": artifact(rules)})
