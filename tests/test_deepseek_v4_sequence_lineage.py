import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.deepseek_v4_sequence_lineage_validate import (
    source_segment,
    validate_audit,
    validate_file,
)

root = Path(__file__).parents[1]
audit_path = root / "results" / "Speck-Paper1" / "deepseek-v4-sequence-lineage-v1.json"


def audit():
    return json.loads(audit_path.read_text(encoding="utf-8"))


def test_deepseek_v4_lineage_retains_architecture_but_supersedes_initial_precision():
    assert validate_file(audit_path) == {
        "status": "valid_initial_precision_superseded",
        "revisions": 2,
        "stable_sequence_segments": 2,
        "precision_source_revision": "current",
        "training_authorized": False,
    }


def test_deepseek_v4_lineage_rejects_initial_precision_authority():
    value = deepcopy(audit())
    value["disposition"]["initial_low_precision_numerical_behavior_authority_retained"] = True
    with pytest.raises(ValueError, match="disposition"):
        validate_audit(value, root)


def test_deepseek_v4_lineage_rejects_local_precision_qualification():
    value = deepcopy(audit())
    value["disposition"]["current_low_precision_behavior_locally_qualified"] = True
    with pytest.raises(ValueError, match="disposition"):
        validate_audit(value, root)


def test_deepseek_v4_lineage_preserves_factorization_and_DAG():
    value = deepcopy(audit())
    value["disposition"]["sequence_factorization_v2_remains_valid"] = False
    with pytest.raises(ValueError, match="disposition"):
        validate_audit(value, root)


def test_source_segment_rejects_missing_markers():
    with pytest.raises(ValueError, match="source marker"):
        source_segment(b"class Other: pass", "class Compressor", "class Gate")
