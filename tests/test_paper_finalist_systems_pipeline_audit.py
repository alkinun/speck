import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.paper_finalist_systems_pipeline_audit_validate import validate_audit, validate_file
from speck.paper_finalist_systems_sampler import GPU_FIELDS
from speck.paper_finalist_systems_telemetry import REQUIRED_FIELDS

root = Path(__file__).parents[1]
audit_path = root / "results" / "Speck-Paper1" / "finalist-systems-pipeline-interface-audit-v1.json"


def audit():
    return json.loads(audit_path.read_text(encoding="utf-8"))


def test_systems_pipeline_audit_is_valid_and_blocked():
    assert validate_file(audit_path) == {
        "status": "valid_blocked",
        "qualified_bridges": 3,
        "blocking_interfaces": 7,
        "execution_authorized": False,
    }


def test_systems_pipeline_reproduces_missing_NVML_memory_producer():
    assert "memory.used" not in GPU_FIELDS
    assert "memory_used_bytes" not in REQUIRED_FIELDS
    nvml = audit()["blocking_interfaces"][1]
    assert nvml["sampler_missing_field"] == "memory.used"
    assert nvml["analyzer_required_field"] == "peak_nvml_used_bytes"


def test_systems_pipeline_audit_rejects_lost_blocker():
    value = deepcopy(audit())
    value["blocking_interfaces"].pop()
    with pytest.raises(ValueError, match="blocker inventory"):
        validate_audit(value, root)


def test_systems_pipeline_audit_rejects_premature_pipeline_qualification():
    value = deepcopy(audit())
    value["decision"]["end_to_end_systems_pipeline_qualified"] = True
    with pytest.raises(ValueError, match="fail-closed decision"):
        validate_audit(value, root)


def test_systems_pipeline_audit_rejects_activation_authority():
    value = deepcopy(audit())
    value["activation_boundary"]["activation_artifact_present"] = True
    with pytest.raises(ValueError, match="activation boundary"):
        validate_audit(value, root)
