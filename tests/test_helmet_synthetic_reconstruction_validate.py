import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.helmet_synthetic_reconstruction_validate import (
    validate_files,
    validate_readiness,
)

root = Path(__file__).parents[1]
readiness_path = (
    root
    / "results"
    / "Speck-Architecture-Promotion-v1"
    / "helmet-synthetic-reconstruction-readiness.json"
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


def test_checked_helmet_synthetic_reconstruction_readiness_is_valid():
    assert validate_files(readiness_path, rights_path, contract_path) == {
        "status": "valid",
        "ruler_task_length_cells": 15,
        "ruler_cases": 1500,
        "official_helmet_reconstruction_qualified": False,
        "evaluation_manifest_changed": False,
    }


def test_reconstruction_rejects_ruler_tokenizer_substitution():
    readiness, rights, contract = _inputs()
    readiness = deepcopy(readiness)
    readiness["existing_ruler_evidence"]["decision"]["helmet_generation_tokenizer_matches"] = True

    with pytest.raises(ValueError, match="RULER non-substitution"):
        validate_readiness(readiness, rights, contract, root)


def test_reconstruction_rejects_ruler_case_hash_drift():
    readiness, rights, contract = _inputs()
    readiness = deepcopy(readiness)
    readiness["existing_ruler_evidence"]["qualifications"][0]["task_sha256"]["niah_multikey_2"] = (
        "0" * 64
    )

    with pytest.raises(ValueError, match="RULER task evidence"):
        validate_readiness(readiness, rights, contract, root)


def test_reconstruction_rejects_json_kv_parity_flip():
    readiness, rights, contract = _inputs()
    readiness = deepcopy(readiness)
    readiness["json_kv_analysis"]["decision"]["exact_helmet_semantics_reconstructable"] = True

    with pytest.raises(ValueError, match="JSON-KV reconstruction"):
        validate_readiness(readiness, rights, contract, root)
