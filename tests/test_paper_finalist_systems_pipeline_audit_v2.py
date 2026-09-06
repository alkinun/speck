import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.paper_finalist_systems_pipeline_audit_v2_validate import validate_audit, validate_file

root = Path(__file__).parents[1]
audit_path = root / "results" / "Speck-Paper1" / "finalist-systems-pipeline-interface-audit-v2.json"


def audit():
    return json.loads(audit_path.read_text(encoding="utf-8"))


def test_v2_pipeline_audit_is_valid_and_live_blocked():
    assert validate_file(audit_path) == {
        "status": "valid_live_blocked",
        "historical_blockers": 7,
        "remaining_activation_gates": 5,
        "activation_present": False,
    }


def test_v2_pipeline_audit_rejects_premature_live_gate_pass():
    value = deepcopy(audit())
    value["remaining_activation_gates"][2]["current_pass"] = True
    with pytest.raises(ValueError, match="activation gates"):
        validate_audit(value, root)


def test_v2_pipeline_audit_rejects_activation_artifact_claim():
    value = deepcopy(audit())
    value["activation"]["artifact_present"] = True
    with pytest.raises(ValueError, match="activation boundary"):
        validate_audit(value, root)


def test_v2_pipeline_audit_preserves_closed_fingerprint_consumer():
    value = audit()
    fingerprint = value["v1_blocker_dispositions"][4]
    assert fingerprint == {
        "id": "paired_batch_fingerprint_consumption",
        "static_state": "closed",
        "remaining": None,
    }
