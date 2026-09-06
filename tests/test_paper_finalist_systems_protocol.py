import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.paper_finalist_systems_protocol_validate import validate_file, validate_protocol

root = Path(__file__).parents[1]
protocol_path = root / "research" / "paper-1" / "finalist_systems_v2.json"


def protocol():
    return json.loads(protocol_path.read_text(encoding="utf-8"))


def test_frozen_finalist_systems_protocol_is_valid_and_blocked():
    assert validate_file(protocol_path) == {
        "status": "valid_frozen_execution_blocked",
        "blocks": 6,
        "trials": 12,
        "measured_steps": 360,
        "balanced_orders": True,
        "activation_gate_pass": False,
    }


def test_finalist_systems_protocol_rejects_unbalanced_order():
    value = deepcopy(protocol())
    value["paired_blocks"][1]["trial_order"] = ["control", "candidate"]
    with pytest.raises(ValueError, match="paired-block order"):
        validate_protocol(value, root)


def test_finalist_systems_protocol_rejects_short_measurement():
    value = deepcopy(protocol())
    value["trial_workload"]["measured_optimizer_steps"] = 5
    with pytest.raises(ValueError, match="trial workload"):
        validate_protocol(value, root)


def test_finalist_systems_protocol_rejects_missing_power_telemetry():
    value = deepcopy(protocol())
    value["telemetry"]["gpu_fields"].remove("power.draw")
    with pytest.raises(ValueError, match="telemetry contract"):
        validate_protocol(value, root)


def test_finalist_systems_protocol_rejects_nonconservative_gap_bound():
    value = deepcopy(protocol())
    value["telemetry"]["gap_upper_bound"] = "use neighboring power samples"
    with pytest.raises(ValueError, match="telemetry contract"):
        validate_protocol(value, root)


def test_finalist_systems_protocol_rejects_weaker_confidence_level():
    value = deepcopy(protocol())
    value["estimands_and_analysis"]["one_sided_confidence"] = 0.95
    with pytest.raises(ValueError, match="estimand or analysis"):
        validate_protocol(value, root)


def test_finalist_systems_protocol_rejects_execution_during_language_sequence():
    value = deepcopy(protocol())
    value["implementation_gate"]["execution_authorized"] = True
    with pytest.raises(ValueError, match="implementation gate"):
        validate_protocol(value, root)
