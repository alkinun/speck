import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.paper_sequence_compression_factorization_v2_validate import (
    validate_design,
    validate_file,
)

root = Path(__file__).parents[1]
design_path = root / "research" / "paper-1" / "sequence_compression_factorization_v2.json"


def design():
    return json.loads(design_path.read_text(encoding="utf-8"))


def test_sequence_factorization_v2_is_corrected_and_training_blocked():
    assert validate_file(design_path) == {
        "status": "valid_joint_factorization_corrected_training_blocked",
        "layer_types": 5,
        "HCA_compressor_arms": 4,
        "experiment_stages": 8,
        "training_authorized": False,
    }


def test_sequence_factorization_rejects_parallel_HCA_CSA_layer():
    value = design()
    value["corrected_layer_grammar"]["HCA_and_CSA_parallel_in_one_layer"] = True
    with pytest.raises(ValueError, match="layer grammar"):
        validate_design(value, root)


def test_sequence_factorization_rejects_cross_family_raw_span_dedup():
    value = design()
    value["entry_identity"]["deduplication"][
        "raw_and_compressed_same_underlying_span_are_duplicates"
    ] = True
    with pytest.raises(ValueError, match="entry identity"):
        validate_design(value, root)


def test_sequence_factorization_requires_exact_dynamic_HCA_arm():
    value = design()
    value["HCA_compressor_isolation_v2"]["arms"].pop()
    with pytest.raises(ValueError, match="HCA compressor"):
        validate_design(value, root)


def test_sequence_factorization_requires_local_before_CSA_schedule_selection():
    value = design()
    value["ordered_experiment_sequence"][3] = "skip local coverage"
    with pytest.raises(ValueError, match="experiment order"):
        validate_design(value, root)


def test_sequence_factorization_rejects_implementation_or_training_authority():
    value = deepcopy(design())
    value["decision"]["implementation_authorized"] = True
    value["decision"]["training_authorized"] = True
    with pytest.raises(ValueError, match="decision"):
        validate_design(value, root)
