"""Validate the source-qualified, training-blocked Stable LatentMoE v2 gate."""

import argparse
import hashlib
import json
from pathlib import Path

from scripts.kimi_k3_primary_source_validate import validate_audit


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
        or design.get("format_version") != 2
        or design.get("gate_id") != "speck-paper-1-stable-latentmoe-readiness-v2"
        or design.get("status")
        != "primary_spec_and_inference_qualified_training_parent_hardware_blocked"
    ):
        raise ValueError("invalid Stable LatentMoE v2 identity")

    predecessor = design.get("predecessor", {})
    v1 = bound_json(root, predecessor, "Stable LatentMoE v1 predecessor changed")
    source_reference = design.get("primary_source_audit", {})
    audit = bound_json(root, source_reference, "Kimi K3 primary-source audit changed")
    validate_audit(audit, root)
    if (
        predecessor.get("preserved_append_only") is not True
        or design.get("required_decomposition_order") != v1.get("required_decomposition_order")
    ):
        raise ValueError("Stable LatentMoE six-stage causal order changed")

    closed = design.get("closed_blockers", {})
    expected_closed = {
        "normalized_LatentMoE_equation_pinned",
        "SiTU_GLU_equation_limits_and_bound_pinned",
        "exact_Quantile_Balancing_equations_pinned",
        "histogram_Quantile_Balancing_estimator_pinned",
        "official_geometry_and_inference_forward_pinned",
        "internal_clean_room_research_license_checked",
    }
    source = design.get("qualified_source_semantics", {})
    histogram = source.get("histogram_Quantile_Balancing", {})
    released = source.get("released_K3_geometry", {})
    if (
        set(closed) != expected_closed
        or any(closed.get(key) is not True for key in expected_closed)
        or source.get("normalized_LatentMoE", {}).get("report_equation") != 11
        or source.get("normalized_LatentMoE", {}).get("normalization_is_a_separate_causal_stage")
        is not True
        or source.get("SiTU_GLU", {}).get("report_equation") != 12
        or source.get("SiTU_GLU", {}).get("gate_beta") != 4
        or source.get("SiTU_GLU", {}).get("up_beta") != 25
        or source.get("SiTU_GLU", {}).get("coordinate_bound") != 100
        or source.get("exact_Quantile_Balancing", {}).get("update_equation") != 14
        or histogram.get("bins") != 1000
        or histogram.get("partition_invariant") is not True
        or not histogram.get("EMA_policy", "").startswith("not part of the initial reference")
        or released.get("routed_experts") != 896
        or released.get("selected_experts") != 16
        or "no released dimension" not in released.get("interpretation", "")
    ):
        raise ValueError("Stable LatentMoE source semantics changed")

    blockers = set(design.get("remaining_hard_blockers", []))
    required_blocker_phrases = (
        "sequence and depth parents",
        "expert-parallel hardware envelope",
        "does not support training",
        "no Quantile Balancing or histogram training updater",
        "no local dropless conventional",
        "no storage/memory budget",
        "no intervention policy",
    )
    if any(not any(phrase in blocker for blocker in blockers) for phrase in required_blocker_phrases):
        raise ValueError("Stable LatentMoE hard blocker removed")

    scope = design.get("isolated_CPU_reference_scope", {})
    forbidden = " ".join(scope.get("forbidden", []))
    if (
        scope.get("authorized") is not True
        or len(scope.get("authorized_primitives", [])) != 4
        or "modifying speck/architecture.py or speck/model.py" not in forbidden
        or "registering or launching any training run" not in forbidden
        or "official inference code as training" not in forbidden
        or "implementing EMA" not in forbidden
        or "production geometry" not in forbidden
        or "skipping conventional MoE" not in forbidden
    ):
        raise ValueError("Stable LatentMoE isolated CPU scope changed")

    active = design.get("active_program_boundary", {})
    program_path = root / active.get("program", "")
    if not program_path.is_file() or file_sha256(program_path) != active.get("sha256"):
        raise ValueError("active Paper 1 program changed")
    program = json.loads(program_path.read_text(encoding="utf-8"))
    pin = program.get("stable_latentmoe_readiness", {})
    if (
        pin.get("contract") != active.get("continues_to_pin")
        or pin.get("sha256") != active.get("predecessor_sha256")
        or active.get("changed") is not False
    ):
        raise ValueError("Stable LatentMoE active-program boundary changed")

    audit_disposition = audit.get("source_disposition", {})
    source_disposition = design.get("primary_source_audit", {})
    decision = design.get("decision", {})
    if (
        source_disposition.get("official_specification_qualified")
        != audit_disposition.get("primary_specification_qualified")
        or source_disposition.get("official_inference_semantics_qualified")
        != audit_disposition.get("official_inference_semantics_qualified")
        or source_disposition.get("official_training_implementation_qualified") is not False
        or source_disposition.get("third_party_implementation_used_as_authority") is not False
        or decision.get("primary_spec_qualified") is not True
        or decision.get("official_inference_semantics_qualified") is not True
        or decision.get("official_training_implementation_qualified") is not False
        or decision.get("sequence_parent_selected") is not False
        or decision.get("depth_parent_selected") is not False
        or decision.get("hardware_envelope_selected") is not False
        or decision.get("conventional_moe_qualified") is not False
        or decision.get("latent_projection_qualified") is not False
        or decision.get("bounded_activation_qualified") is not False
        or decision.get("balancing_qualified") is not False
        or decision.get("expert_geometry_selected") is not False
        or decision.get("isolated_CPU_reference_implementation_authorized") is not True
        or decision.get("local_model_integration_authorized") is not False
        or decision.get("training_authorized") is not False
        or decision.get("promotion_authority") is not False
    ):
        raise ValueError("Stable LatentMoE v2 disposition changed")
    return {
        "status": "valid_CPU_reference_authorized_training_blocked",
        "causal_stages": len(design["required_decomposition_order"]),
        "isolated_CPU_primitives": len(scope["authorized_primitives"]),
        "remaining_hard_blockers": len(blockers),
        "training_authorized": False,
    }


def validate_file(path):
    path = Path(path).resolve()
    return validate_design(json.loads(path.read_text(encoding="utf-8")), path.parents[2])


def main(argv=None):
    report = validate_file(arguments(argv).design)
    print(
        "Stable LatentMoE readiness v2: "
        f"{report['status']} ({report['causal_stages']} stages, "
        f"{report['isolated_CPU_primitives']} CPU primitives)"
    )


if __name__ == "__main__":
    main()
