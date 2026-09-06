import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.paper_sequence_axis_convergence_validate import validate_design, validate_file

root = Path(__file__).parents[1]
design_path = root / "research" / "paper-1" / "sequence_axis_convergence_v1.json"


def design():
    return json.loads(design_path.read_text(encoding="utf-8"))


def test_sequence_axis_convergence_is_acyclic_and_training_blocked():
    assert validate_file(design_path) == {
        "status": "valid_dependency_DAG_ready_for_qualification",
        "nodes": 11,
        "topological_nodes": 11,
        "known_run_envelopes": 8,
        "training_authorized": False,
    }


def test_sequence_axis_convergence_rejects_cycle():
    value = design()
    value["DAG_nodes"][0]["depends_on"] = ["I1_three_axis_interaction"]
    with pytest.raises(ValueError, match="cycle"):
        validate_design(value, root)


def test_sequence_axis_convergence_rejects_conditional_parent_reuse():
    value = design()
    value["detected_order_conflicts"]["cache_v2_conditional_parent"] = "any parent"
    with pytest.raises(ValueError, match="conflict resolution"):
        validate_design(value, root)


def test_sequence_axis_convergence_rejects_ratio_output_reuse_after_compression():
    value = design()
    value["DAG_nodes"][7]["requirement"] = "reuse ratio-v2 results"
    with pytest.raises(ValueError, match="node contract"):
        validate_design(value, root)


def test_sequence_axis_convergence_rejects_concurrent_single_GPU_stages():
    value = design()
    value["execution_rules"]["S2a_and_S2b_may_run_concurrently"] = True
    with pytest.raises(ValueError, match="execution rule"):
        validate_design(value, root)


def test_sequence_axis_convergence_rejects_early_materialization_or_training():
    value = deepcopy(design())
    value["decision"]["descendant_configs_materialized"] = True
    value["decision"]["training_authorized"] = True
    with pytest.raises(ValueError, match="decision"):
        validate_design(value, root)
