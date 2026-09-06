import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.paper_attnres_readiness_v2_validate import validate_design, validate_file

root = Path(__file__).parents[1]
design_path = root / "research" / "paper-1" / "attnres_readiness_v2.json"


def design():
    return json.loads(design_path.read_text(encoding="utf-8"))


def test_attnres_v2_corrects_boundaries_and_blocks_training():
    assert validate_file(design_path) == {
        "status": "valid_partition_corrected_training_blocked",
        "residual_modules": 40,
        "N8_module_sizes": [4, 6, 4, 6, 4, 6, 4, 6],
        "initial_arms": 4,
        "training_authorized": False,
    }


def test_attnres_v2_rejects_odd_module_boundary():
    value = design()
    value["corrected_block_families"]["N8"]["module_boundaries"][1] = 5
    with pytest.raises(ValueError, match="block family"):
        validate_design(value, root)


def test_attnres_v2_rejects_five_module_arm():
    value = design()
    value["initial_isolation_arms"][3]["residual_module_sizes"] = [5] * 8
    with pytest.raises(ValueError, match="initial isolation"):
        validate_design(value, root)


def test_attnres_v2_requires_exact_zero_query_initialization():
    value = design()
    value["source_exact_equations"]["Full_AttnRes"]["pseudo_query_initialization"] = (
        "generic normal"
    )
    with pytest.raises(ValueError, match="source equations"):
        validate_design(value, root)


def test_attnres_v2_rejects_K3_training_authority():
    value = design()
    value["primary_source_audit"]["official_training_implementation_qualified"] = True
    with pytest.raises(ValueError, match="source boundary"):
        validate_design(value, root)


def test_attnres_v2_rejects_implementation_or_training_authority():
    value = deepcopy(design())
    value["decision"]["reference_implementation_authorized"] = True
    value["decision"]["training_authorized"] = True
    with pytest.raises(ValueError, match="decision"):
        validate_design(value, root)
