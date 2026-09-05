from copy import deepcopy
from pathlib import Path

import pytest

from scripts.paper_finalist_continue import ordered_runs
from scripts.paper_finalist_launch_qualify import (
    arguments,
    finalist_units,
    validate_initial_state,
)


def initial_state():
    return {
        "finalist_evidence": {
            "status": "qualified_unexecuted",
            "control_results": [],
            "candidate_results": [],
            "time_to_quality_target": None,
            "analysis_result": None,
            "next_run": ordered_runs()[0],
        }
    }


def test_finalist_launch_qualification_arguments():
    args = arguments(["--output", "result.json"])
    assert args.output == Path("result.json")


def test_finalist_launch_qualification_requires_exact_empty_state():
    program = initial_state()
    assert validate_initial_state(program) == ordered_runs()[0]
    changed = deepcopy(program)
    changed["finalist_evidence"]["control_results"] = [{"pair": 0}]
    with pytest.raises(ValueError, match="empty initial state"):
        validate_initial_state(changed)


def test_finalist_launch_qualification_covers_every_unit_type():
    units = finalist_units()
    assert len(units) == 48
    assert len(set(units)) == 48
    assert {Path(unit).suffix for unit in units} == {".service", ".timer", ".path"}
