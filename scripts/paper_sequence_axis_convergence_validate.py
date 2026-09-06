"""Validate the pre-results Paper 1 sequence/depth/width dependency DAG."""

import argparse
import hashlib
import json
from pathlib import Path


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


def validate_graph(nodes):
    identifiers = [node.get("id") for node in nodes]
    if len(identifiers) != len(set(identifiers)) or any(not value for value in identifiers):
        raise ValueError("sequence convergence DAG node ids must be unique")
    known = set(identifiers)
    dependencies = {node["id"]: set(node.get("depends_on", [])) for node in nodes}
    if any(not values <= known for values in dependencies.values()):
        raise ValueError("sequence convergence DAG references an unknown node")
    remaining = {key: set(value) for key, value in dependencies.items()}
    resolved = []
    while remaining:
        ready = sorted(key for key, value in remaining.items() if not value)
        if not ready:
            raise ValueError("sequence convergence DAG contains a cycle")
        resolved.extend(ready)
        for key in ready:
            remaining.pop(key)
        for value in remaining.values():
            value.difference_update(ready)
    return resolved


def validate_inputs(inputs, root):
    expected = {
        "NoPE_factorial",
        "cache_representation",
        "base_ratio",
        "recurrent_gate",
        "compressed_attention",
        "depth_successor",
        "width_successor",
        "three_axis_interaction",
    }
    if set(inputs) != expected:
        raise ValueError("sequence convergence inputs changed")
    for reference in inputs.values():
        for path_key, hash_key in (
            ("design", "design_sha256"),
            ("qualification", "qualification_sha256"),
        ):
            path = root / reference.get(path_key, "")
            if not path.is_file() or file_sha256(path) != reference.get(hash_key):
                raise ValueError("sequence convergence input hash changed")


def validate_design(design, root):
    if (
        design.get("format") != "speck_sequence_axis_convergence_gate"
        or design.get("format_version") != 1
        or design.get("gate_id") != "speck-paper-1-sequence-axis-convergence-v1"
        or design.get("status")
        != "pre_results_dependency_DAG_frozen_active_finalist_and_parent_selection_blocked"
    ):
        raise ValueError("invalid sequence axis convergence identity")
    validate_inputs(design.get("inputs", {}), root)

    conflicts = design.get("detected_order_conflicts", {})
    if (
        "selected before" not in conflicts.get("compression_v2_stage_one", "")
        or "after compressed operator" not in conflicts.get("compression_v2_stage_eight", "")
        or "distinct questions" not in conflicts.get("resolution", "")
        or conflicts.get("compression_v2_factorization_content_valid") is not True
        or conflicts.get("compression_v2_order_authority") is not False
        or conflicts.get("cache_v2_conditional_parent")
        != "KDA, NoPE, five global GQA3 slots, sigmoid gate"
        or conflicts.get("ratio_v2_conditional_parent")
        != "KDA, NoPE, GQA3, sigmoid gate, standard residual"
        or "neither v2 may be registered unchanged" not in conflicts.get(
            "conditional_design_rule", ""
        )
    ):
        raise ValueError("sequence convergence conflict resolution changed")

    nodes = design.get("DAG_nodes", [])
    order = validate_graph(nodes)
    by_id = {node["id"]: node for node in nodes}
    expected_ids = [
        "F0_active_finalist",
        "S1_mixer_position_factorial",
        "S2a_cache_representation",
        "S2b_base_refresh_ratio",
        "S3_cache_ratio_integration",
        "S4_recurrent_gate",
        "S5_compressed_attention",
        "S6_compressed_schedule_refresh_revalidation",
        "D1_depth_axis",
        "W1_width_axis",
        "I1_three_axis_interaction",
    ]
    if [node.get("id") for node in nodes] != expected_ids:
        raise ValueError("sequence convergence DAG order changed")
    expected_dependencies = {
        "F0_active_finalist": [],
        "S1_mixer_position_factorial": ["F0_active_finalist"],
        "S2a_cache_representation": ["S1_mixer_position_factorial"],
        "S2b_base_refresh_ratio": ["S1_mixer_position_factorial"],
        "S3_cache_ratio_integration": ["S2a_cache_representation", "S2b_base_refresh_ratio"],
        "S4_recurrent_gate": ["S3_cache_ratio_integration"],
        "S5_compressed_attention": ["S4_recurrent_gate"],
        "S6_compressed_schedule_refresh_revalidation": ["S5_compressed_attention"],
        "D1_depth_axis": ["S6_compressed_schedule_refresh_revalidation"],
        "W1_width_axis": ["S6_compressed_schedule_refresh_revalidation", "D1_depth_axis"],
        "I1_three_axis_interaction": [
            "S6_compressed_schedule_refresh_revalidation",
            "D1_depth_axis",
            "W1_width_axis",
        ],
    }
    if any(by_id[key].get("depends_on") != value for key, value in expected_dependencies.items()):
        raise ValueError("sequence convergence DAG dependency changed")
    if (
        by_id["F0_active_finalist"].get("status") != "running_event_driven"
        or by_id["S1_mixer_position_factorial"].get("known_new_runs") != 18
        or by_id["S2a_cache_representation"].get("known_runs")
        != {"discovery": 12, "finalist_if_authorized": 24}
        or "only under selected KDA/NoPE" not in by_id["S2a_cache_representation"].get(
            "conditional_identity", ""
        )
        or by_id["S2b_base_refresh_ratio"].get("known_runs")
        != {"discovery": 15, "finalist_if_authorized": 18}
        or "only under selected KDA/NoPE" not in by_id["S2b_base_refresh_ratio"].get(
            "conditional_identity", ""
        )
        or by_id["S3_cache_ratio_integration"].get("materialized") is not False
        or "difference-in-differences" not in by_id["S3_cache_ratio_integration"].get(
            "requirement", ""
        )
        or "6 if exact" not in by_id["S4_recurrent_gate"].get("known_new_runs", "")
        or "re-confirm" not in by_id["S4_recurrent_gate"].get("changed_gate_rule", "")
        or len(by_id["S5_compressed_attention"].get("internal_order", [])) != 7
        or "do not re-select base ratio" not in by_id["S5_compressed_attention"].get(
            "corrected_entry", ""
        )
        or by_id["S6_compressed_schedule_refresh_revalidation"].get("materialized") is not False
        or "do not reuse ratio-v2 outputs" not in by_id[
            "S6_compressed_schedule_refresh_revalidation"
        ].get("requirement", "")
        or by_id["I1_three_axis_interaction"].get("known_discovery_runs") != 24
    ):
        raise ValueError("sequence convergence node contract changed")

    rules = design.get("execution_rules", {})
    if (
        rules.get("topological_order_required") is not True
        or rules.get("S2a_and_S2b_may_run_concurrently") is not False
        or "single consumer GPU" not in rules.get("reason_for_serial_S2", "")
        or rules.get("one_stage_registered_at_a_time") is not True
        or rules.get("every_future_artifact_frozen_before_its_first_output") is not True
        or rules.get("no_result_can_rewrite_upstream_design") is not True
        or rules.get("no_automatic_full_grid_materialization") is not True
        or rules.get("exact_parent_hash_required_at_every_edge") is not True
        or rules.get("active_finalist_outputs_may_only_enter_S1_through_frozen_reuse_contract")
        is not True
    ):
        raise ValueError("sequence convergence execution rule changed")

    envelopes = design.get("known_run_envelopes_not_authority", {})
    expected_counts = {
        "active_finalist_total": 12,
        "NoPE_new": 18,
        "cache_discovery": 12,
        "cache_finalist_if_authorized": 24,
        "base_ratio_discovery": 15,
        "base_ratio_finalist_if_authorized": 18,
        "recurrent_gate_new": "6 or 12",
        "three_axis_discovery": 24,
    }
    if any(envelopes.get(key) != value for key, value in expected_counts.items()) or (
        "not a batch launch authorization" not in envelopes.get("rule", "")
    ):
        raise ValueError("sequence convergence run envelope changed")

    stop = design.get("pre_results_stop_rule", {})
    forbidden = " ".join(stop.get("forbidden", []))
    if (
        stop.get("sequence_source_and_factorization_work_complete") is not True
        or stop.get("no_further_sequence_source_or_synthetic_expansion_before_F0") is not True
        or len(stop.get("allowed", [])) != 3
        or "S1 or descendant" not in forbidden
        or "descendant configs early" not in forbidden
        or "non-KDA or positional-mismatched parent" not in forbidden
        or "ratio-v2 output as compressed-schedule" not in forbidden
        or "concurrently" not in forbidden
    ):
        raise ValueError("sequence convergence stop rule changed")

    active = design.get("active_program_boundary", {})
    program_path = root / active.get("program", "")
    if not program_path.is_file() or file_sha256(program_path) != active.get("sha256"):
        raise ValueError("active Paper 1 program changed")
    if active.get("changed") is not False or active.get("convergence_gate_registered") is not False:
        raise ValueError("sequence convergence active-program boundary changed")
    decision = design.get("decision", {})
    if (
        decision.get("dependency_DAG_qualified") is not False
        or decision.get("sequence_parent_selected") is not False
        or decision.get("descendant_configs_materialized") is not False
        or decision.get("implementation_authorized") is not False
        or decision.get("training_authorized") is not False
        or decision.get("promotion_authority") is not False
    ):
        raise ValueError("sequence convergence decision changed")
    return {
        "status": "valid_dependency_DAG_ready_for_qualification",
        "nodes": len(nodes),
        "topological_nodes": len(order),
        "known_run_envelopes": len(expected_counts),
        "training_authorized": False,
    }


def validate_file(path):
    path = Path(path).resolve()
    return validate_design(json.loads(path.read_text(encoding="utf-8")), path.parents[2])


def main(argv=None):
    report = validate_file(arguments(argv).design)
    print(
        "Sequence axis convergence: "
        f"{report['status']} ({report['nodes']} nodes, "
        f"training={str(report['training_authorized']).lower()})"
    )


if __name__ == "__main__":
    main()
