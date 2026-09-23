"""Readiness records must agree with the declared mixture and the retained-data closeout."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_source_readiness", ROOT / "experiments/main-data/check_source_readiness.py"
)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)

MATRIX = ROOT / "experiments/main-data/source-readiness.json"


def test_the_recorded_matrix_agrees_with_the_current_mixture():
    result = checker.validate(MATRIX)

    assert result["admitted_sources"] == 0
    assert result["status"] == "candidate_evidence_only_no_training_admission"


def test_a_bound_divided_by_a_share_the_mixture_no_longer_has_is_rejected(tmp_path):
    matrix = json.loads(MATRIX.read_text())
    bound = matrix["horizon_accounting"]["one_pass_constraints"][0]
    # Self-consistent arithmetic at a stale share: exactly the drift that passed before.
    bound["declared_share_percent"] = 30
    bound["maximum_total_exposure_tokens_before_exclusions"] = (
        bound["numerator_tokens"] * 100
    ) // 30
    drifted = tmp_path / "source-readiness.json"
    drifted.write_text(json.dumps(matrix))

    with pytest.raises(ValueError, match="drifts from the mixture"):
        checker.validate(drifted)


def test_a_bound_naming_no_bank_is_rejected(tmp_path):
    matrix = json.loads(MATRIX.read_text())
    matrix["horizon_accounting"]["one_pass_constraints"][0].pop("bank")
    drifted = tmp_path / "source-readiness.json"
    drifted.write_text(json.dumps(matrix))

    with pytest.raises(ValueError, match="names no declared bank"):
        checker.validate(drifted)


def test_an_inventory_that_drifts_from_its_cited_closeout_is_rejected(tmp_path):
    matrix = json.loads(MATRIX.read_text())
    pes2o = next(source for source in matrix["sources"] if source["id"] == "pes2o_v3")
    # The superseded stock-v1 counts this entry once carried.
    pes2o["inventory"].update(documents=55787, retained_tokens=403558463)
    drifted = tmp_path / "source-readiness.json"
    drifted.write_text(json.dumps(matrix))

    with pytest.raises(ValueError, match="drifts from the closeout"):
        checker.validate(drifted)


def _drifted(tmp_path, matrix):
    path = tmp_path / "source-readiness.json"
    path.write_text(json.dumps(matrix))
    return path


def test_a_selection_that_differs_from_the_signed_acceptance_is_rejected(tmp_path):
    matrix = json.loads(MATRIX.read_text())
    next(s for s in matrix["sources"] if s["id"] == "checked_code")["selected"] = True

    with pytest.raises(ValueError, match="signed source-use acceptance"):
        checker.validate(_drifted(tmp_path, matrix))


@pytest.mark.parametrize("value", ["bound_value", "complete"])
def test_a_manifest_field_outside_the_vocabulary_is_rejected(tmp_path, value):
    matrix = json.loads(MATRIX.read_text())
    matrix["sources"][0]["manifest_fields"]["stage"] = value

    with pytest.raises(ValueError, match="manifest fields drift"):
        checker.validate(_drifted(tmp_path, matrix))


@pytest.mark.parametrize("value", [1, -1, True, "0"])
def test_a_source_cannot_claim_eligible_tokens(tmp_path, value):
    matrix = json.loads(MATRIX.read_text())
    stack_edu = next(s for s in matrix["sources"] if s["id"] == "stack_edu")
    stack_edu["inventory"]["eligible_unique_tokens_established"] = value

    with pytest.raises(ValueError, match="cannot claim eligible tokens"):
        checker.validate(_drifted(tmp_path, matrix))
