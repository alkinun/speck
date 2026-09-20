"""Planning checks must reject drift even when the overall allocation still balances."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_program_plan", ROOT / "experiments/main-data/check_program_plan.py"
)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


@pytest.fixture
def plan():
    data = json.loads((ROOT / "experiments/main-data/plan.json").read_text())
    # Unit tests exercise budgeting independently of the retained evidence tree.
    # Dedicated tests below provide their own checksum and missing-file cases.
    data["input_receipts"] = {}
    return data


def validate_plan(tmp_path, plan):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return checker.validate(path)


def test_current_design_remains_non_authorizing(tmp_path, plan):
    result = validate_plan(tmp_path, plan)
    assert result["reservation_gpu_hours"] == 5000
    assert result["training_authority"] is False


def test_balanced_but_conflicting_research_components_rejected(tmp_path, plan):
    budget = plan["compute"]["proposed_research_breakdown_gpu_hours"]["pretraining"]
    budget["confirmation_training"] += 30
    budget["preparation_evaluation_recovery"] -= 30
    with pytest.raises(ValueError, match="research components drift"):
        validate_plan(tmp_path, plan)


def test_cross_stage_reallocation_requires_packet_revision(tmp_path, plan):
    budget = plan["compute"]["data_experiments_breakdown_gpu_hours"]
    budget["pretraining"] -= 10
    budget["mid_training"] += 10
    with pytest.raises(ValueError, match="study packet does not match"):
        validate_plan(tmp_path, plan)


def test_stale_screening_arm_count_rejected(tmp_path, plan):
    plan["pretraining_data_study"]["proposed_screening"]["max_arms"] = 3
    with pytest.raises(ValueError, match="run caps drift"):
        validate_plan(tmp_path, plan)


def test_missing_receipt_rejected(tmp_path, plan):
    plan["input_receipts"] = {str(tmp_path / "missing.json"): "0" * 64}
    with pytest.raises(ValueError, match="missing input receipt"):
        validate_plan(tmp_path, plan)


def test_changed_receipt_rejected(tmp_path, plan):
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}")
    plan["input_receipts"] = {str(receipt): "0" * 64}
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_plan(tmp_path, plan)


def test_matching_receipt_counted(tmp_path, plan):
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}")
    plan["input_receipts"] = {str(receipt): checker._sha256(receipt)}
    assert validate_plan(tmp_path, plan)["input_receipts_checked"] == 1


def test_stale_readiness_horizon_rejected(tmp_path, plan, monkeypatch):
    original_load = checker._load

    def stale_load(path):
        value = original_load(path)
        if path == "experiments/main-data/source-readiness.json":
            value["horizon_accounting"]["working_target"]["total_exposure_tokens"] = 100_000_000_000
        return value

    monkeypatch.setattr(checker, "_load", stale_load)
    with pytest.raises(ValueError, match="source-readiness horizon"):
        validate_plan(tmp_path, plan)
