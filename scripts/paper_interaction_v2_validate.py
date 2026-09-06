"""Validate exact 2^3 interaction contrasts and the corrected readiness design."""

import argparse
import hashlib
import json
import math
from pathlib import Path

CELLS = ("000", "100", "010", "001", "110", "101", "011", "111")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("design", type=Path)
    return parser.parse_args(argv)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cube_contrasts(values, coefficients):
    if set(values) != set(CELLS) or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in values.values()
    ):
        raise ValueError("interaction cube values are incomplete or non-finite")
    if any(set(weights) != set(CELLS) for weights in coefficients.values()):
        raise ValueError("interaction coefficient vector is incomplete")
    return {
        name: sum(weights[cell] * values[cell] for cell in CELLS)
        for name, weights in coefficients.items()
    }


def no_hiding_guard(upper_bounds, names, margin):
    if set(upper_bounds) != set(names):
        raise ValueError("conditional interaction upper-bound set is incomplete")
    return all(upper_bounds[name] <= margin for name in names)


def expected_coefficients():
    def vector(positive=(), negative=(), scale=1):
        return {
            cell: scale if cell in positive else -scale if cell in negative else 0 for cell in CELLS
        }

    conditional = {
        "SD_W0": vector(("000", "110"), ("100", "010")),
        "SD_W1": vector(("001", "111"), ("101", "011")),
        "SW_D0": vector(("000", "101"), ("100", "001")),
        "SW_D1": vector(("010", "111"), ("110", "011")),
        "DW_S0": vector(("000", "011"), ("010", "001")),
        "DW_S1": vector(("100", "111"), ("110", "101")),
    }

    def average(left, right):
        return {cell: (conditional[left][cell] + conditional[right][cell]) / 2 for cell in CELLS}

    return {
        "S": {cell: (0.25 if cell[0] == "1" else -0.25) for cell in CELLS},
        "D": {cell: (0.25 if cell[1] == "1" else -0.25) for cell in CELLS},
        "W": {cell: (0.25 if cell[2] == "1" else -0.25) for cell in CELLS},
        "SD": average("SD_W0", "SD_W1"),
        "SW": average("SW_D0", "SW_D1"),
        "DW": average("DW_S0", "DW_S1"),
        "SDW": {cell: conditional["SD_W1"][cell] - conditional["SD_W0"][cell] for cell in CELLS},
        **conditional,
        "S_from_full": vector(("111",), ("011",)),
        "D_from_full": vector(("111",), ("101",)),
        "W_from_full": vector(("111",), ("110",)),
    }


def validate_design(design, root):
    if (
        design.get("format") != "speck_interaction_readiness_gate"
        or design.get("format_version") != 2
        or design.get("status") != "estimands_corrected_axes_unselected_training_blocked"
    ):
        raise ValueError("invalid interaction readiness v2 identity")
    predecessor = design.get("predecessor", {})
    predecessor_path = root / predecessor.get("path", "")
    if (
        not predecessor_path.is_file()
        or file_sha256(predecessor_path) != predecessor.get("sha256")
        or predecessor.get("preserved") is not True
    ):
        raise ValueError("interaction readiness v2 predecessor changed")
    for name, reference in design.get("updated_sequence_prerequisites", {}).items():
        path = root / reference.get("path", "")
        if not path.is_file() or file_sha256(path) != reference.get("sha256"):
            raise ValueError(f"interaction readiness v2 {name} prerequisite changed")
    cube = design.get("discovery_cube", {})
    coefficients = design.get("contrast_coefficients", {})
    expected_names = {
        "S",
        "D",
        "W",
        "SD",
        "SW",
        "DW",
        "SDW",
        "SD_W0",
        "SD_W1",
        "SW_D0",
        "SW_D1",
        "DW_S0",
        "DW_S1",
        "S_from_full",
        "D_from_full",
        "W_from_full",
    }
    if (
        cube.get("cell_order") != list(CELLS)
        or cube.get("paired_cells") != 3
        or cube.get("model_runs") != 24
        or cube.get("fixed_sample") is not True
        or cube.get("interim_cell_dropping") is not False
        or cube.get("promotion_authority") is not False
        or set(coefficients) != expected_names
        or coefficients != expected_coefficients()
    ):
        raise ValueError("interaction readiness v2 cube or contrast inventory changed")
    additive = {cell: 1 + 2 * int(cell[0]) + 3 * int(cell[1]) + 4 * int(cell[2]) for cell in CELLS}
    contrasts = cube_contrasts(additive, coefficients)
    if (
        contrasts["S"] != 2
        or contrasts["D"] != 3
        or contrasts["W"] != 4
        or any(contrasts[name] != 0 for name in ("SD", "SW", "DW", "SDW"))
        or coefficients["S_from_full"]
        != {
            "000": 0,
            "100": 0,
            "010": 0,
            "001": 0,
            "110": 0,
            "101": 0,
            "011": -1,
            "111": 1,
        }
    ):
        raise ValueError("interaction readiness v2 coefficient semantics changed")
    inference = design.get("inference", {})
    registration = design.get("registration_boundary", {})
    if (
        inference.get("paired_cells") != 3
        or inference.get("student_t_critical_df_2_one_sided_95") != 2.919985580353724
        or inference.get("student_t_critical_df_2_two_sided_95") != 4.302652729696142
        or inference.get("primary_Holm_family") != ["SD", "SW", "DW", "SDW"]
        or inference.get("conditional_no_hiding_guards")
        != ["SD_W0", "SD_W1", "SW_D0", "SW_D1", "DW_S0", "DW_S1"]
        or inference.get("conditional_diagnostics_published") is not True
        or inference.get("averaged_interaction_cannot_hide_conditional_harm") is not True
        or inference.get("failure_to_detect_interaction_is_not_independence") is not True
        or registration.get("active_experiment_program_changed") is not False
        or registration.get("v1_future_cube_training_authorized") is not False
        or registration.get("v2_registered") is not False
        or registration.get("axes_selected") is not False
        or registration.get("cube_materialized") is not False
        or registration.get("training_authorized") is not False
        or registration.get("promotion_authority") is not False
    ):
        raise ValueError("interaction readiness v2 inference or decision changed")
    return {
        "status": "valid_training_blocked",
        "cells": len(CELLS),
        "contrasts": len(coefficients),
        "Holm_family": len(inference["primary_Holm_family"]),
        "conditional_guards": len(inference["conditional_no_hiding_guards"]),
        "training_authorized": False,
    }


def validate_file(path):
    path = Path(path).resolve()
    return validate_design(json.loads(path.read_text(encoding="utf-8")), path.parents[2])


def main(argv=None):
    report = validate_file(arguments(argv).design)
    print(
        "Interaction readiness v2: "
        f"{report['status']} ({report['cells']} cells, {report['Holm_family']} Holm, "
        f"{report['conditional_guards']} guards)"
    )


if __name__ == "__main__":
    main()
