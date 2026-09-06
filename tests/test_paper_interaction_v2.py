import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.paper_interaction_v2_validate import (
    CELLS,
    cube_contrasts,
    no_hiding_guard,
    validate_design,
    validate_file,
)

root = Path(__file__).parents[1]
design_path = root / "research" / "paper-1" / "interaction_readiness_v2.json"


def design():
    return json.loads(design_path.read_text(encoding="utf-8"))


def test_interaction_v2_validates_exact_estimands_and_remains_blocked():
    assert validate_file(design_path) == {
        "status": "valid_training_blocked",
        "cells": 8,
        "contrasts": 16,
        "Holm_family": 4,
        "conditional_guards": 6,
        "training_authorized": False,
    }


def test_additive_cube_has_only_main_effects():
    value = design()
    outcomes = {cell: 1 + 2 * int(cell[0]) + 3 * int(cell[1]) + 4 * int(cell[2]) for cell in CELLS}
    contrasts = cube_contrasts(outcomes, value["contrast_coefficients"])
    assert (contrasts["S"], contrasts["D"], contrasts["W"]) == (2, 3, 4)
    assert all(contrasts[name] == 0 for name in ("SD", "SW", "DW", "SDW"))


def test_pairwise_primary_is_average_of_both_conditionals():
    value = design()
    outcomes = {cell: 5 * int(cell[0]) * int(cell[1]) for cell in CELLS}
    contrasts = cube_contrasts(outcomes, value["contrast_coefficients"])
    assert contrasts["SD_W0"] == 5
    assert contrasts["SD_W1"] == 5
    assert contrasts["SD"] == 5
    assert contrasts["SDW"] == 0


def test_conditional_guard_catches_harm_hidden_by_zero_average():
    value = design()
    outcomes = {cell: 0.0 for cell in CELLS}
    outcomes["110"] = 0.02
    outcomes["111"] = -0.02
    contrasts = cube_contrasts(outcomes, value["contrast_coefficients"])
    assert contrasts["SD"] == 0
    assert contrasts["SD_W0"] == 0.02
    assert contrasts["SD_W1"] == -0.02
    names = value["inference"]["conditional_no_hiding_guards"]
    bounds = {name: -0.02 for name in names}
    bounds["SD_W0"] = 0.02
    assert no_hiding_guard(bounds, names, 0.01) is False


def test_interaction_v2_rejects_changed_Holm_family():
    value = design()
    value["inference"]["primary_Holm_family"].remove("SDW")
    with pytest.raises(ValueError, match="inference or decision"):
        validate_design(value, root)


def test_interaction_v2_rejects_incomplete_coefficient_vector():
    value = design()
    value["contrast_coefficients"]["SD"].pop("111")
    with pytest.raises(ValueError, match="cube or contrast inventory"):
        validate_design(value, root)


def test_interaction_v2_rejects_complete_but_changed_coefficient_vector():
    value = design()
    value["contrast_coefficients"]["SD"]["111"] = 0.25
    with pytest.raises(ValueError, match="cube or contrast inventory"):
        validate_design(value, root)


def test_interaction_v2_rejects_training_or_registration_authority():
    value = deepcopy(design())
    value["registration_boundary"]["training_authorized"] = True
    with pytest.raises(ValueError, match="inference or decision"):
        validate_design(value, root)
