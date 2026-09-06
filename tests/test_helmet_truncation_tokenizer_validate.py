import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.helmet_truncation_tokenizer_validate import validate_files, validate_readiness

root = Path(__file__).parents[1]
readiness_path = (
    root
    / "results"
    / "Speck-Architecture-Promotion-v1"
    / "helmet-truncation-tokenizer-readiness.json"
)
protocol_path = (
    root / "research" / "architecture-promotion-v1" / "helmet_runtime_dependencies_v1.json"
)
audit_path = (
    root / "results" / "Speck-Architecture-Promotion-v1" / "helmet-runtime-dependency-audit.json"
)
synthetic_path = (
    root
    / "results"
    / "Speck-Architecture-Promotion-v1"
    / "helmet-synthetic-reconstruction-readiness.json"
)


def _inputs():
    return tuple(
        json.loads(path.read_text(encoding="utf-8"))
        for path in (readiness_path, protocol_path, audit_path, synthetic_path)
    )


def test_checked_helmet_truncation_tokenizer_readiness_is_valid():
    assert validate_files(readiness_path, protocol_path, audit_path, synthetic_path) == {
        "status": "valid",
        "runtime_entries": 25,
        "ruler_generation_cells": 15,
        "replacement_oracle_available": False,
        "evaluation_manifest_changed": False,
    }


def test_tokenizer_readiness_rejects_local_payload_claim():
    readiness, protocol, audit, synthetic = _inputs()
    readiness = deepcopy(readiness)
    readiness["inputs"]["llama2"]["local_tokenizer_payload_found"] = True

    with pytest.raises(ValueError, match="authority boundary"):
        validate_readiness(readiness, protocol, audit, synthetic, root)


def test_tokenizer_readiness_rejects_fixture_only_equivalence():
    readiness, protocol, audit, synthetic = _inputs()
    readiness = deepcopy(readiness)
    readiness["replacement_gate"]["fixture_only_equivalence_sufficient"] = True

    with pytest.raises(ValueError, match="replacement gate"):
        validate_readiness(readiness, protocol, audit, synthetic, root)


def test_tokenizer_readiness_rejects_acquisition_authority_flip():
    readiness, protocol, audit, synthetic = _inputs()
    readiness = deepcopy(readiness)
    readiness["decision"]["tokenizer_payload_acquisition_authorized"] = True

    with pytest.raises(ValueError, match="fail-closed decision"):
        validate_readiness(readiness, protocol, audit, synthetic, root)
