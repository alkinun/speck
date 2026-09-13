import hashlib
import json
from pathlib import Path

import pytest

from speck.tokenization.pilot import _validate_run
from tests.reference import historical_repository

ROOT = historical_repository()
RESULT = ROOT / "results/data/tokenizer-pilot-mistral-seed42-20260913.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.mark.evidence
def test_mistral_screen_completion_is_bound_and_keeps_D5_closed():
    result = json.loads(RESULT.read_text())

    assert result["status"] == "mistral_seed42_screen_arm_complete"
    assert sha256(ROOT / result["execution"]["path"]) == result["execution"]["sha256"]
    for identity in result["interruptions"]["checked_records"]:
        assert sha256(ROOT / identity["path"]) == identity["sha256"]
    assert result["metrics"]["learning_curve_points"] == 20
    assert result["metrics"]["fixed_document_macro_bpb"] == 1.0954040025770306
    assert result["gates"]["analyzer_schema"] == "pass"
    assert result["gates"]["D5_audit_openings"] == 0
    assert result["authority"]["confirmation_execution"] is False
    assert result["authority"]["D5_opening"] is False


@pytest.mark.integration
def test_mistral_runtime_result_matches_and_validates_when_available():
    result = json.loads(RESULT.read_text())
    path = Path(result["runtime"]["result"]["path"])
    if not path.is_file():
        pytest.skip("requires maintainer-local completed Mistral screen arm")
    assert path.stat().st_size == result["runtime"]["result"]["bytes"]
    assert sha256(path) == result["runtime"]["result"]["sha256"]
    run = json.loads(path.read_text())
    plan = json.loads((ROOT / "research/flagship/tokenizer_pilot_plan_v10.json").read_text())
    _validate_run(run, plan)
    assert len(run["learning_curve"]) == 20

    summary = Path(result["runtime"]["summary"]["path"])
    assert summary.stat().st_size == result["runtime"]["summary"]["bytes"]
    assert sha256(summary) == result["runtime"]["summary"]["sha256"]
