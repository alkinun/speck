import json
from copy import deepcopy
from pathlib import Path

import pytest

from speck.research import validate_research_contract

root = Path(__file__).parents[1]
contract = root / "research" / "architecture-promotion-v1"


def test_checked_architecture_promotion_contract_is_valid():
    report = validate_research_contract(contract)
    assert report == {
        "policy_id": "architecture-promotion-v1",
        "contract_files": [
            "policy.json",
            "cost_envelopes.json",
            "evaluation_manifest.json",
            "evidence_matrix.json",
        ],
        "replication_stages": 6,
        "training_profiles": 2,
        "serving_profiles": 3,
        "internal_evaluation_suites": 4,
        "external_evaluation_suites": 3,
        "evidence_components": 10,
        "status": "valid",
    }


def test_contract_rejects_equivalence_ratio_drift(tmp_path):
    destination = tmp_path / "contract"
    destination.mkdir()
    for source in contract.iterdir():
        value = json.loads(source.read_text(encoding="utf-8"))
        if source.name == "policy.json":
            value = deepcopy(value)
            value["statistical_contract"]["language_loss"]["perplexity_ratio_at_margin"] = 1.0
        (destination / source.name).write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="perplexity ratio"):
        validate_research_contract(destination)
