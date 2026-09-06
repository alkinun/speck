import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.helmet_model_judge_validate import validate_files, validate_readiness

root = Path(__file__).parents[1]
readiness_path = (
    root / "results" / "Speck-Architecture-Promotion-v1" / "helmet-model-judge-readiness.json"
)
audit_path = (
    root / "results" / "Speck-Architecture-Promotion-v1" / "helmet-runtime-dependency-audit.json"
)
protocol_path = (
    root / "research" / "architecture-promotion-v1" / "helmet_runtime_dependencies_v1.json"
)
contract_path = root / "research" / "architecture-promotion-v1" / "external" / "helmet.json"


def _inputs():
    return tuple(
        json.loads(path.read_text(encoding="utf-8"))
        for path in (readiness_path, protocol_path, audit_path, contract_path)
    )


def test_checked_helmet_model_judge_readiness_is_valid():
    assert validate_files(readiness_path, protocol_path, audit_path, contract_path) == {
        "status": "valid",
        "runtime_entries": 15,
        "seed": 42,
        "repeatability_trials": 1,
        "candidate_scoring_authorized": False,
        "evaluation_manifest_changed": False,
    }


def test_model_judge_rejects_seed_null_regression():
    readiness, protocol, audit, contract = _inputs()
    readiness = deepcopy(readiness)
    readiness["request_contract"]["seed"] = None

    with pytest.raises(ValueError, match="request contract"):
        validate_readiness(readiness, protocol, audit, contract, root)


def test_model_judge_rejects_discarded_fingerprint_claim():
    readiness, protocol, audit, contract = _inputs()
    readiness = deepcopy(readiness)
    readiness["request_contract"]["system_fingerprint_preserved_in_scored_json"] = True

    with pytest.raises(ValueError, match="request contract"):
        validate_readiness(readiness, protocol, audit, contract, root)


def test_model_judge_rejects_success_only_denominator():
    readiness, protocol, audit, contract = _inputs()
    readiness = deepcopy(readiness)
    readiness["replacement_gate"]["successful_only_denominator_authorized"] = True

    with pytest.raises(ValueError, match="replacement gate"):
        validate_readiness(readiness, protocol, audit, contract, root)


def test_model_judge_rejects_api_authority_flip():
    readiness, protocol, audit, contract = _inputs()
    readiness = deepcopy(readiness)
    readiness["decision"]["api_execution_authorized"] = True

    with pytest.raises(ValueError, match="fail-closed decision"):
        validate_readiness(readiness, protocol, audit, contract, root)
