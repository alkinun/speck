import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.helmet_archive_local_rights_validate import validate_audit, validate_files

root = Path(__file__).parents[1]
audit_path = (
    root / "results" / "Speck-Architecture-Promotion-v1" / "helmet-archive-local-rights-audit.json"
)
inspection_path = (
    root / "results" / "Speck-Architecture-Promotion-v1" / "helmet-archive-inspection.json"
)
contract_path = root / "research" / "architecture-promotion-v1" / "external" / "helmet.json"


def _inputs():
    return tuple(
        json.loads(path.read_text(encoding="utf-8"))
        for path in (audit_path, inspection_path, contract_path)
    )


def test_checked_helmet_archive_local_rights_audit_is_valid():
    assert validate_files(audit_path, inspection_path, contract_path) == {
        "status": "valid",
        "families": 5,
        "paths": 52,
        "extraction_qualified_families": 0,
        "archive_member_identity_sha256": (
            "23d404071444d52686c917ab9827b604594f2faa23502d1e9b45fa9c558bbd10"
        ),
    }


def test_rights_audit_rejects_family_count_drift():
    audit, inspection, contract = _inputs()
    audit = deepcopy(audit)
    audit["families"][0]["paths"] += 1

    with pytest.raises(ValueError, match="family disposition"):
        validate_audit(
            audit,
            inspection,
            contract,
            audit["archive"]["inspection_sha256"],
        )


def test_rights_audit_rejects_authorization_flip():
    audit, inspection, contract = _inputs()
    audit = deepcopy(audit)
    audit["families"][0]["extraction_authorized"] = True

    with pytest.raises(ValueError, match="family disposition"):
        validate_audit(
            audit,
            inspection,
            contract,
            audit["archive"]["inspection_sha256"],
        )


def test_rights_audit_rejects_inspection_path_drift():
    audit, inspection, contract = _inputs()
    inspection = deepcopy(inspection)
    inspection["inspection"]["declared_local_paths"].pop()

    with pytest.raises(ValueError, match="path-family accounting"):
        validate_audit(
            audit,
            inspection,
            contract,
            audit["archive"]["inspection_sha256"],
        )
