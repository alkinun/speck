import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.attnres_primary_source_validate import (
    expected_boundaries,
    validate_audit,
    validate_file,
)

root = Path(__file__).parents[1]
audit_path = root / "results" / "Speck-Paper1" / "attnres-primary-source-audit-v1.json"


def audit():
    return json.loads(audit_path.read_text(encoding="utf-8"))


def test_attnres_primary_source_audit_requires_v2_partition_correction():
    assert validate_file(audit_path) == {
        "status": "valid_v2_correction_authorized_training_blocked",
        "residual_modules": 40,
        "N8_module_sizes": [4, 6, 4, 6, 4, 6, 4, 6],
        "official_training_qualified": False,
    }


def test_attnres_boundaries_are_balanced_and_logical_layer_aligned():
    for summaries in (4, 8, 12):
        boundaries, sizes = expected_boundaries(20, summaries)
        assert boundaries[0] == 0 and boundaries[-1] == 20
        assert all(boundary * 2 % 2 == 0 for boundary in boundaries)
        assert sum(sizes) == 40
        assert max(sizes) - min(sizes) <= 2


def test_attnres_audit_rejects_five_module_blocks():
    value = deepcopy(audit())
    value["corrected_Speck_boundaries"]["N8"]["residual_module_sizes"] = [5] * 8
    with pytest.raises(ValueError, match="boundary arithmetic"):
        validate_audit(value, root)


def test_attnres_audit_rejects_released_zero_initialization_claim():
    value = deepcopy(audit())
    value["official_K3_code_boundary"][
        "special_zero_initialization_for_AttnRes_projections_present"
    ] = True
    with pytest.raises(ValueError, match="disposition"):
        validate_audit(value, root)


def test_attnres_audit_rejects_implementation_or_training_authority():
    value = deepcopy(audit())
    value["source_disposition"]["local_reference_implementation_authorized"] = True
    value["source_disposition"]["training_authorized"] = True
    with pytest.raises(ValueError, match="disposition"):
        validate_audit(value, root)
