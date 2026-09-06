from copy import deepcopy
from pathlib import Path

import pytest

from scripts.quantile_balancing_tie_validate import load_object, validate_protocol

root = Path(__file__).parents[1]
protocol_path = root / "research" / "paper-1" / "quantile_balancing_tie_readiness_v2.json"


def protocol():
    return load_object(protocol_path)


def test_QB_tie_protocol_is_frozen_and_training_blocked():
    contract = validate_protocol(protocol(), root)
    validation = contract["validation"]

    assert validation["cases"] == 256
    assert validation["candidate_gate"]["policy"] == "interval_midpoint"
    assert contract["base"]["required_future_training_gate"][
        "source_upper_histogram_and_midpoint_arms_required"
    ] is True
    assert protocol()["boundary"]["training_authorized"] is False


def test_QB_tie_protocol_rejects_post_hoc_seed_change():
    value = deepcopy(protocol())
    value["fresh_unseen_validation"]["seeds_per_shape"] = "0 through 63 inclusive"
    with pytest.raises(ValueError, match="v2 validation contract changed"):
        validate_protocol(value, root)


def test_QB_tie_protocol_rejects_training_authority():
    value = deepcopy(protocol())
    value["boundary"]["training_authorized"] = True
    with pytest.raises(ValueError, match="contract changed"):
        validate_protocol(value, root)
