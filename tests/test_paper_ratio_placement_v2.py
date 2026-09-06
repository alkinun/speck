import json
from pathlib import Path

import pytest

from scripts.paper_ratio_placement_v2_validate import EXPECTED, validate_design, validate_file

root = Path(__file__).parents[1]
design_path = root / "research" / "paper-1" / "ratio_placement_readiness_v2.json"


def design():
    return json.loads(design_path.read_text(encoding="utf-8"))


def test_ratio_placement_v2_recomputes_five_geometries_and_is_blocked():
    assert validate_file(design_path) == {
        "status": "valid_training_blocked",
        "raw_arms": 3,
        "matched_sensitivities": 2,
        "discovery_runs": 15,
        "training_authorized": False,
    }


def test_ratio_placement_v2_requires_all_raw_arms():
    value = design()
    value["raw_selection_arms"].pop()
    with pytest.raises(ValueError, match="arm inventory"):
        validate_design(value, root, EXPECTED)


def test_ratio_placement_v2_rejects_matched_selection_authority():
    value = design()
    value["parameter_matched_sensitivities"][0]["promotion_authority"] = True
    with pytest.raises(ValueError, match="gained authority"):
        validate_design(value, root, EXPECTED)


def test_ratio_placement_v2_rejects_changed_raw_selection_family():
    value = design()
    value["causal_contrasts"]["raw_selection_family"].reverse()
    with pytest.raises(ValueError, match="analysis or decision"):
        validate_design(value, root, EXPECTED)


def test_ratio_placement_v2_rejects_placement_before_raw_selection():
    value = design()
    value["placement_successor"]["status"] = "frozen"
    with pytest.raises(ValueError, match="analysis or decision"):
        validate_design(value, root, EXPECTED)


def test_ratio_placement_v2_rejects_training_authority():
    value = design()
    value["decision"]["training_authorized"] = True
    with pytest.raises(ValueError, match="analysis or decision"):
        validate_design(value, root, EXPECTED)
