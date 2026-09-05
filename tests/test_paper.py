import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest

from speck.paper import validate_paper_program

root = Path(__file__).parents[1]
program = root / "research" / "paper-1"


def test_checked_paper_program_is_valid_and_pretraining_is_blocked():
    assert validate_paper_program(program) == {
        "paper_id": "speck-paper-1",
        "status": "valid_hypotheses_only",
        "program_files": [
            "README.md",
            "claims.json",
            "baseline_matrix.json",
            "experiment_program.json",
            "paper_outline.md",
            "reference_audit.md",
            "reporting_checklist.md",
        ],
        "claims": 7,
        "non_claims": 8,
        "scales": 6,
        "axes": 3,
        "historical_baseline_arms": 5,
        "planned_primary_baseline_arms": 2,
        "proxy_confirmation_pairs": 3,
        "paper_scale_pretraining": "blocked",
    }


def test_paper_program_rejects_an_unknown_axis_claim(tmp_path):
    copied = tmp_path / "paper-1"
    shutil.copytree(program, copied)
    path = copied / "experiment_program.json"
    value = deepcopy(json.loads(path.read_text(encoding="utf-8")))
    value["axes"][0]["claim_ids"].append("C99")
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid claims"):
        validate_paper_program(copied, repository_root=root)
