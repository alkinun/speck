import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.paper_finalist_continue import ordered_runs
from scripts.paper_finalist_launch_qualify import (
    arguments,
    finalist_units,
    validate_initial_state,
)
from speck.paper import _validate_finalist_live_launch


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


def test_live_gate_is_exact_while_empty_and_historical_after_progress():
    root = Path(__file__).parents[1]
    program = json.loads(
        (root / "research/paper-1/experiment_program.json").read_text(encoding="utf-8")
    )
    reference = program["finalist_live_launch"]
    empty = program["finalist_evidence"]
    _validate_finalist_live_launch(reference, root, "speck-paper-1", empty)
    progressed = deepcopy(empty)
    progressed["status"] = "in_progress"
    progressed["control_results"] = [{"pair": 0}]
    _validate_finalist_live_launch(reference, root, "speck-paper-1", progressed)
    invalid_empty = deepcopy(empty)
    invalid_empty["next_run"] = "wrong"
    with pytest.raises(ValueError, match="incomplete"):
        _validate_finalist_live_launch(reference, root, "speck-paper-1", invalid_empty)
