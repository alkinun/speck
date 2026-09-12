import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
RESULT = ROOT / "results/data/tokenizer-pilot-screen-materialization-20260912.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_checked_screen_materialization_binds_plan_code_and_closed_authority():
    result = json.loads(RESULT.read_text())

    assert result["status"] == "three_screen_run_manifests_complete_trainer_and_evaluator_pending"
    assert sha256(ROOT / result["plan"]["path"]) == result["plan"]["sha256"]
    for path, digest in result["implementation"].values():
        assert sha256(ROOT / path) == digest
    assert len(result["runtime"]["runs"]) == 3
    assert result["gates"]["model_outputs_created"] == 0
    assert result["gates"]["D5_audit_openings"] == 0
    assert result["authority"] == {
        "screen_execution": True,
        "confirmation_execution": False,
        "D5_opening": False,
        "final_selection": False,
        "flagship_training": False,
    }


def test_runtime_screen_manifests_match_checked_record_when_available():
    result = json.loads(RESULT.read_text())
    manifest = Path(result["runtime"]["manifest"]["path"])
    if not manifest.is_file():
        pytest.skip("requires maintainer-local tokenizer pilot run materialization")

    assert sha256(manifest) == result["runtime"]["manifest"]["sha256"]
    runtime = json.loads(manifest.read_text())
    assert runtime["plan_fingerprint"] == result["plan"]["plan_fingerprint"]
    for run in result["runtime"]["runs"]:
        path = Path(result["runtime"]["directory"]) / run["path"]
        assert sha256(path) == run["sha256"]
        value = json.loads(path.read_text())
        assert value["run_fingerprint"] == run["run_fingerprint"]
        assert value["status"] == "materialized_not_started"
