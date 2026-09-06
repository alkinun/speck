"""Validate corrected logical-layer-aligned Attention Residuals readiness v2."""

import argparse
import hashlib
import json
from pathlib import Path

from scripts.attnres_primary_source_validate import expected_boundaries, validate_audit


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
        design.get("format") != "speck_attnres_readiness_gate"
        or design.get("format_version") != 2
        or design.get("gate_id") != "speck-paper-1-attnres-readiness-v2"
        or design.get("status")
        != "source_corrected_logical_boundaries_frozen_sequence_parent_implementation_training_blocked"
    ):
        raise ValueError("invalid AttnRes readiness v2 identity")
    predecessor = design.get("predecessor", {})
    v1 = bound_json(root, predecessor, "AttnRes v1 predecessor changed")
    source_reference = design.get("primary_source_audit", {})
    audit = bound_json(root, source_reference, "AttnRes primary-source audit changed")
    validate_audit(audit, root)
    if (
        predecessor.get("preserved_append_only") is not True
        or predecessor.get("partition_authority") is not False
        or source_reference.get("primary_equations_qualified") is not True
        or source_reference.get("K3_Block_inference_semantics_qualified") is not True
        or source_reference.get("from_scratch_initialization_qualified_by_K3_code") is not False
        or source_reference.get("official_training_implementation_qualified") is not False
    ):
        raise ValueError("AttnRes v2 source boundary changed")

    graph = design.get("module_graph", {})
    if (
        graph.get("logical_transformer_blocks") != 20
        or graph.get("residual_modules_per_logical_block") != 2
        or graph.get("residual_modules") != 40
        or "never between" not in graph.get("block_boundary_constraint", "")
        or v1.get("module_graph", {}).get("residual_modules") != graph.get("residual_modules")
    ):
        raise ValueError("AttnRes v2 module graph changed")

    equations = design.get("source_exact_equations", {})
    full = equations.get("Full_AttnRes", {})
    block = equations.get("Block_AttnRes", {})
    if (
        full.get("source_equations") != [2, 3, 4]
        or full.get("pseudo_query_initialization") != "exact zero"
        or full.get("initial_weights") != "uniform over available sources"
        or full.get("weights") != "single softmax over depth sources"
        or full.get("values") != "unnormalized sources"
        or block.get("source_equations") != [5, 6]
        or "two-phase" not in block.get("optimized_path", "")
    ):
        raise ValueError("AttnRes v2 source equations changed")

    families = design.get("corrected_block_families", {})
    for summaries, name in ((4, "N4"), (8, "N8"), (12, "N12")):
        logical, sizes = expected_boundaries(20, summaries)
        modules = [2 * boundary for boundary in logical]
        value = families.get(name, {})
        if (
            value.get("logical_boundaries") != logical
            or value.get("module_boundaries") != modules
            or value.get("module_sizes") != sizes
            or any(boundary % 2 for boundary in modules)
        ):
            raise ValueError("AttnRes v2 corrected block family changed")
    if families.get("N8", {}).get("maximum_sources") != 9:
        raise ValueError("AttnRes v2 N8 source count changed")

    arms = design.get("initial_isolation_arms", [])
    block_arm = next((arm for arm in arms if arm.get("id") == "block_attnres_8"), {})
    if (
        [arm.get("id") for arm in arms]
        != ["standard_prenorm", "static_depth_weights", "full_attnres", "block_attnres_8"]
        or block_arm.get("logical_block_sizes") != [2, 3, 2, 3, 2, 3, 2, 3]
        or block_arm.get("residual_module_sizes") != [4, 6, 4, 6, 4, 6, 4, 6]
        or block_arm.get("maximum_sources") != 9
    ):
        raise ValueError("AttnRes v2 initial isolation changed")

    code = design.get("official_code_disposition", {})
    not_authoritative = " ".join(code.get("not_authoritative", []))
    if (
        "checkpoint inference only" not in code.get("use", "")
        or len(code.get("confirmed", [])) != 5
        or "generic linear initialization is nonzero" not in not_authoritative
        or "no full residual-module mode" not in not_authoritative
        or "sparse MoE rejects training" not in not_authoritative
        or "neither path is released" not in not_authoritative
    ):
        raise ValueError("AttnRes v2 official-code disposition changed")

    correctness = design.get("correctness_before_training", [])
    prerequisites = design.get("ordered_activation_prerequisites", [])
    successor = design.get("block_count_successor", {})
    if (
        len(correctness) != 10
        or not any("exact-zero" in item for item in correctness)
        or not any("even and aligned" in item for item in correctness)
        or len(prerequisites) != 5
        or prerequisites[0] != "active finalist sequence completes and is accepted"
        or prerequisites[1] != "one conservative or selected sequence parent is frozen"
        or successor.get("summary_counts") != [4, 8, 12]
        or "floor(i*20/N)" not in successor.get("boundary_rule", "")
    ):
        raise ValueError("AttnRes v2 correctness or activation order changed")

    active = design.get("active_program_boundary", {})
    program_path = root / active.get("program", "")
    if not program_path.is_file() or file_sha256(program_path) != active.get("sha256"):
        raise ValueError("active Paper 1 program changed")
    program = json.loads(program_path.read_text(encoding="utf-8"))
    if (
        program.get("attnres_readiness", {}).get("contract") != active.get("continues_to_pin")
        or active.get("changed") is not False
    ):
        raise ValueError("AttnRes v2 active-program boundary changed")

    decision = design.get("decision", {})
    if (
        decision.get("v1_partition_valid") is not False
        or decision.get("v2_partition_corrected") is not True
        or decision.get("residual_selected") is not False
        or decision.get("block_count_selected") is not False
        or decision.get("depth_width_geometry_selected") is not False
        or decision.get("sequence_parent_selected") is not False
        or decision.get("reference_implementation_authorized") is not False
        or decision.get("model_integration_authorized") is not False
        or decision.get("training_authorized") is not False
        or decision.get("promotion_authority") is not False
    ):
        raise ValueError("AttnRes v2 decision changed")
    return {
        "status": "valid_partition_corrected_training_blocked",
        "residual_modules": 40,
        "N8_module_sizes": families["N8"]["module_sizes"],
        "initial_arms": len(arms),
        "training_authorized": False,
    }


def validate_file(path):
    path = Path(path).resolve()
    return validate_design(json.loads(path.read_text(encoding="utf-8")), path.parents[2])


def main(argv=None):
    report = validate_file(arguments(argv).design)
    print(
        "Attention Residuals readiness v2: "
        f"{report['status']} (N8={report['N8_module_sizes']}, "
        f"training={str(report['training_authorized']).lower()})"
    )


if __name__ == "__main__":
    main()
