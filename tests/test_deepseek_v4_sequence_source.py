import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.deepseek_v4_sequence_source_validate import (
    schedule_counts,
    validate_audit,
    validate_file,
)

root = Path(__file__).parents[1]
audit_path = (
    root / "results" / "Speck-Paper1" / "deepseek-v4-sequence-primary-source-audit-v1.json"
)


def audit():
    return json.loads(audit_path.read_text(encoding="utf-8"))


def test_deepseek_v4_sequence_audit_requires_three_v2_corrections():
    assert validate_file(audit_path) == {
        "status": "valid_three_v2_corrections_authorized_training_blocked",
        "Flash_schedule": {"SWA": 2, "CSA": 21, "HCA": 20},
        "local_gates_requiring_correction": 3,
        "training_authorized": False,
    }


def test_schedule_counts_excludes_trailing_MTP_entry():
    config = {"num_hidden_layers": 3, "compress_ratios": [0, 4, 128, 0]}
    assert schedule_counts(config) == {"SWA": 1, "CSA": 1, "HCA": 1}


def test_v4_audit_rejects_parallel_HCA_CSA_claim():
    value = deepcopy(audit())
    value["official_architecture_factorization"]["same_layer_HCA_plus_CSA_branches"] = True
    with pytest.raises(ValueError, match="sequence semantics"):
        validate_audit(value, root)


def test_v4_audit_rejects_raw_compressed_span_deduplication():
    value = deepcopy(audit())
    value["official_local_and_normalization_semantics"][
        "deduplicate_by_underlying_raw_source_span"
    ] = True
    with pytest.raises(ValueError, match="sequence semantics"):
        validate_audit(value, root)


def test_v4_audit_rejects_arbitrary_chunked_prefill_claim():
    value = deepcopy(audit())
    value["official_causal_state_semantics"]["arbitrary_nonzero_start_chunked_prefill"] = True
    with pytest.raises(ValueError, match="sequence semantics"):
        validate_audit(value, root)


def test_v4_audit_rejects_local_implementation_or_training_authority():
    value = deepcopy(audit())
    value["source_disposition"]["local_reference_implementation_authorized"] = True
    value["source_disposition"]["training_authorized"] = True
    with pytest.raises(ValueError, match="source disposition"):
        validate_audit(value, root)
