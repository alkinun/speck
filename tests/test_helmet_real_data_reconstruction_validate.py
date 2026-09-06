import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.helmet_real_data_reconstruction_validate import (
    validate_files,
    validate_readiness,
)

root = Path(__file__).parents[1]
readiness_path = (
    root
    / "results"
    / "Speck-Architecture-Promotion-v1"
    / "helmet-real-data-reconstruction-readiness.json"
)
rights_path = (
    root / "results" / "Speck-Architecture-Promotion-v1" / "helmet-archive-local-rights-audit.json"
)
contract_path = root / "research" / "architecture-promotion-v1" / "external" / "helmet.json"


def _inputs():
    return tuple(
        json.loads(path.read_text(encoding="utf-8"))
        for path in (readiness_path, rights_path, contract_path)
    )


def test_checked_helmet_real_data_reconstruction_readiness_is_valid():
    assert validate_files(readiness_path, rights_path, contract_path) == {
        "status": "valid",
        "families": 3,
        "archive_paths": 32,
        "exact_official_reconstruction_qualified": False,
        "evaluation_manifest_changed": False,
    }


def test_real_data_reconstruction_rejects_rag_parity_flip():
    readiness, rights, contract = _inputs()
    readiness = deepcopy(readiness)
    readiness["families"][0]["exact_semantics_reconstructable"] = True

    with pytest.raises(ValueError, match="family disposition"):
        validate_readiness(readiness, rights, contract, root)


def test_real_data_reconstruction_rejects_alce_depth_collapse():
    readiness, rights, contract = _inputs()
    readiness = deepcopy(readiness)
    readiness["families"][2]["source_mismatch"]["helmet_archive_gtr_depth"] = 100

    with pytest.raises(ValueError, match="citation reconstruction"):
        validate_readiness(readiness, rights, contract, root)


def test_real_data_reconstruction_rejects_source_acquisition_flip():
    readiness, rights, contract = _inputs()
    readiness = deepcopy(readiness)
    readiness["decision"]["source_payload_acquisition_authorized"] = True

    with pytest.raises(ValueError, match="fail-closed decision"):
        validate_readiness(readiness, rights, contract, root)
