import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.paper_finalist_systems_audit_validate import validate_audit, validate_file

root = Path(__file__).parents[1]
audit_path = root / "results" / "Speck-Paper1" / "finalist-systems-measurement-audit-v1.json"


def load_audit():
    return json.loads(audit_path.read_text(encoding="utf-8"))


def test_checked_finalist_systems_audit_is_valid():
    assert validate_file(audit_path) == {
        "status": "valid",
        "runs": 12,
        "forecast_steady_gpu_hours": 121.2303502872009,
        "energy_measured": False,
        "causal_systems_claim_authorized": False,
        "language_sequence_authorized": True,
    }


def test_finalist_systems_audit_rejects_energy_claim():
    audit = deepcopy(load_audit())
    audit["decision"]["energy_claim_authorized"] = True
    with pytest.raises(ValueError, match="fail-closed decision"):
        validate_audit(audit, root)


def test_finalist_systems_audit_rejects_interleaving_claim():
    audit = deepcopy(load_audit())
    audit["execution_design"]["arms_interleaved"] = True
    with pytest.raises(ValueError, match="execution design"):
        validate_audit(audit, root)


def test_finalist_systems_audit_rejects_persisted_successor_telemetry_claim():
    audit = deepcopy(load_audit())
    audit["execution_design"]["successor_live_gate_values_persisted"] = True
    with pytest.raises(ValueError, match="execution design"):
        validate_audit(audit, root)
