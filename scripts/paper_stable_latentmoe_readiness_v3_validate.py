"""Validate the pre-results-complete, activation-blocked width readiness v3."""

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


def bound_json(root, reference, message):
    path = root / reference.get("path", "")
    if not path.is_file() or file_sha256(path) != reference.get("sha256"):
        raise ValueError(message)
    return json.loads(path.read_text(encoding="utf-8"))


def validate_design(design, root):
    if (
        design.get("format") != "speck_stable_latentmoe_readiness_gate"
        or design.get("format_version") != 3
        or design.get("gate_id") != "speck-paper-1-stable-latentmoe-readiness-v3"
        or design.get("status") != "pre_results_width_references_complete_activation_blocked"
    ):
        raise ValueError("invalid Stable LatentMoE readiness v3 identity")
    predecessor = design.get("predecessor", {})
    v2 = bound_json(root, predecessor, "Stable LatentMoE v2 predecessor changed")
    bound_json(root, predecessor.get("qualification", {}), "Stable LatentMoE v2 qualification changed")
    if (
        predecessor.get("preserved_append_only") is not True
        or design.get("required_decomposition_order") != v2.get("required_decomposition_order")
    ):
        raise ValueError("Stable LatentMoE v3 causal order changed")

    evidence = design.get("new_evidence", {})
    cpu = bound_json(root, evidence.get("four_isolated_CPU_primitives", {}), "CPU reference qualification changed")
    tie_v1 = bound_json(root, evidence.get("QB_tie_gate_v1", {}), "QB tie v1 changed")
    tie_v2_reference = evidence.get("QB_tie_gate_v2", {})
    tie_v2 = bound_json(root, tie_v2_reference, "QB tie v2 changed")
    tie_qualification = bound_json(
        root, tie_v2_reference.get("qualification", {}), "QB tie v2 qualification changed"
    )
    conventional = bound_json(
        root, evidence.get("conventional_MoE_primary_source", {}), "conventional MoE source changed"
    )
    if (
        evidence.get("four_isolated_CPU_primitives", {}).get("qualified") is not True
        or evidence.get("four_isolated_CPU_primitives", {}).get("composition_qualified") is not False
        or cpu.get("decision", {}).get("training_authorized") is not False
        or evidence.get("QB_tie_gate_v1", {}).get("result_created") is not False
        or tie_v1.get("format_version") != 1
        or tie_v2.get("format_version") != 2
        or evidence.get("QB_tie_gate_v2", {}).get("fixed_score_CPU_midpoint_qualified") is not True
        or evidence.get("QB_tie_gate_v2", {}).get("training_tie_policy_qualified") is not False
        or tie_qualification.get("candidate_gate", {}).get("passed") is not True
        or tie_qualification.get("decision", {}).get("training_tie_policy_selected") is not False
        or evidence.get("conventional_MoE_primary_source", {}).get("primary_equations_qualified")
        is not True
        or evidence.get("conventional_MoE_primary_source", {}).get(
            "single_device_per_layer_dropless_training_semantics_qualified"
        )
        is not True
        or evidence.get("conventional_MoE_primary_source", {}).get(
            "expert_parallel_training_implementation_qualified"
        )
        is not False
        or conventional.get("source_disposition", {}).get(
            "local_conventional_MoE_implementation_authorized"
        )
        is not False
    ):
        raise ValueError("Stable LatentMoE v3 evidence disposition changed")

    knowledge = design.get("qualified_pre_results_knowledge", {})
    expected_true = {
        "K3_normalized_LatentMoE_SiTU_exact_QB_and_histogram_specification",
        "K3_official_inference_semantics",
        "isolated_float64_CPU_equation_oracles",
        "fixed_score_midpoint_tie_safe_CPU_oracle",
        "DeepSeekMoE_conventional_fine_grained_shared_expert_equations",
        "DeepSeekMoE_single_device_per_layer_dropless_training_path",
    }
    expected_false = {
        "K3_official_training_implementation",
        "fixed_score_midpoint_is_a_training_policy",
        "expert_parallel_training_path",
        "Speck_width_architecture_selected",
    }
    if (
        any(knowledge.get(key) is not True for key in expected_true)
        or any(knowledge.get(key) is not False for key in expected_false)
    ):
        raise ValueError("Stable LatentMoE v3 knowledge boundary changed")

    prerequisites = design.get("activation_prerequisites", {})
    prerequisite_states = {key: value for key, value in prerequisites.items() if key != "all_required"}
    if (
        len(prerequisite_states) != 7
        or any(value is not False for value in prerequisite_states.values())
        or prerequisites.get("all_required") is not True
    ):
        raise ValueError("Stable LatentMoE v3 activation prerequisite changed")

    semantics = design.get("first_stage_semantics_to_freeze_after_activation", {})
    required_semantics = {
        "placement",
        "shared_experts",
        "routed_expert_count_and_topk",
        "router_score_family",
        "selected_weight_normalization",
        "ties",
        "capacity",
        "dispatch",
        "expert_initialization",
        "auxiliary_balance",
        "accounting",
    }
    if (
        set(semantics) != required_semantics
        or "exactly one" not in semantics.get("shared_experts", "")
        or "explicit false" not in semantics.get("selected_weight_normalization", "")
        or "not hardware authority" not in semantics.get("ties", "")
        or "dropless" not in semantics.get("capacity", "")
    ):
        raise ValueError("Stable LatentMoE first-stage freeze list changed")

    stop = design.get("stop_rule", {})
    forbidden = " ".join(stop.get("forbidden_before_activation", []))
    if (
        stop.get("no_further_width_source_or_synthetic_expansion_before_activation") is not True
        or len(stop.get("allowed_maintenance", [])) != 3
        or "active architecture/model paths" not in forbidden
        or "dispatcher or kernel" not in forbidden
        or "training job" not in forbidden
        or "fixed-score midpoint as a training policy" not in forbidden
        or "register v3" not in forbidden
    ):
        raise ValueError("Stable LatentMoE v3 stop rule changed")

    active = design.get("active_program_boundary", {})
    program_path = root / active.get("program", "")
    if not program_path.is_file() or file_sha256(program_path) != active.get("sha256"):
        raise ValueError("active Paper 1 program changed")
    program = json.loads(program_path.read_text(encoding="utf-8"))
    if (
        program.get("stable_latentmoe_readiness", {}).get("contract")
        != active.get("continues_to_pin")
        or active.get("changed") is not False
    ):
        raise ValueError("Stable LatentMoE v3 active-program boundary changed")

    decision = design.get("decision", {})
    if (
        decision.get("pre_results_width_reference_work_complete") is not True
        or decision.get("source_gate_complete") is not True
        or decision.get("isolated_CPU_oracles_complete") is not True
        or decision.get("conventional_MoE_implementation_authorized") is not False
        or decision.get("primitive_composition_authorized") is not False
        or decision.get("model_integration_authorized") is not False
        or decision.get("training_authorized") is not False
        or decision.get("promotion_authority") is not False
    ):
        raise ValueError("Stable LatentMoE v3 decision changed")
    return {
        "status": "valid_pre_results_width_complete_activation_blocked",
        "causal_stages": len(design["required_decomposition_order"]),
        "activation_prerequisites": len(prerequisite_states),
        "first_stage_semantics": len(semantics),
        "implementation_authorized": False,
        "training_authorized": False,
    }


def validate_file(path):
    path = Path(path).resolve()
    return validate_design(json.loads(path.read_text(encoding="utf-8")), path.parents[2])


def main(argv=None):
    report = validate_file(arguments(argv).design)
    print(
        "Stable LatentMoE readiness v3: "
        f"{report['status']} ({report['activation_prerequisites']} prerequisites, "
        f"training={str(report['training_authorized']).lower()})"
    )


if __name__ == "__main__":
    main()
