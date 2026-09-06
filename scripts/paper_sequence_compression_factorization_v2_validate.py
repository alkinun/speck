"""Validate corrected joint HCA/CSA/raw-local sequence factorization v2."""

import argparse
import hashlib
import json
from pathlib import Path

from scripts.deepseek_v4_sequence_source_validate import validate_audit


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
        design.get("format") != "speck_sequence_compression_factorization_gate"
        or design.get("format_version") != 2
        or design.get("gate_id") != "speck-paper-1-sequence-compression-factorization-v2"
        or design.get("status")
        != "source_corrected_joint_factorization_frozen_parents_implementation_training_blocked"
    ):
        raise ValueError("invalid sequence compression factorization v2 identity")
    predecessors = design.get("predecessors", {})
    expected_predecessors = {
        "HCA_v1": "speck_hca_readiness_gate",
        "CSA_v1": "speck_csa_readiness_gate",
        "raw_local_v1": "speck_raw_local_readiness_gate",
    }
    for name, format_value in expected_predecessors.items():
        reference = predecessors.get(name, {})
        predecessor = bound_json(root, reference, f"sequence predecessor changed: {name}")
        if (
            predecessor.get("format") != format_value
            or reference.get("preserved_append_only") is not True
            or reference.get("factorization_authority") is not False
        ):
            raise ValueError("sequence predecessor authority changed")
    source_reference = design.get("primary_source_audit", {})
    source = bound_json(root, source_reference, "DeepSeek-V4 source audit changed")
    validate_audit(source, root)
    if (
        source_reference.get("primary_equations_qualified") is not True
        or source_reference.get("official_inference_semantics_qualified") is not True
        or source_reference.get("official_training_implementation_qualified") is not False
    ):
        raise ValueError("sequence source boundary changed")

    grammar = design.get("corrected_layer_grammar", {})
    if (
        grammar.get("available_types")
        != ["exact_global", "HCA_core", "HCA_local", "CSA_core", "CSA_local"]
        or grammar.get("one_compressed_type_per_layer") is not True
        or grammar.get("HCA_and_CSA_parallel_in_one_layer") is not False
        or "mechanism-isolation diagnostic only" not in grammar.get("HCA_core", "")
        or "source-faithful HCA" not in grammar.get("HCA_local", "")
        or "mechanism-isolation diagnostic only" not in grammar.get("CSA_core", "")
        or "source-faithful CSA" not in grammar.get("CSA_local", "")
        or "different slots" not in grammar.get("schedule_rule", "")
        or grammar.get("published_V4_schedule_selected_for_Speck") is not False
    ):
        raise ValueError("sequence layer grammar changed")

    identity = design.get("entry_identity", {})
    hca = identity.get("HCA", {})
    csa = identity.get("CSA", {})
    dedup = identity.get("deduplication", {})
    if (
        identity.get("identity_tuple")
        != [
            "representation_family",
            "emitted_entry_index",
            "ordered_source_span_set",
            "projection_and_precision_identity",
        ]
        or hca.get("overlap") is not False
        or hca.get("stride") != "m'"
        or csa.get("stride") != "m"
        or csa.get("support_after_first") != "2m"
        or csa.get("overlap") is not True
        or dedup.get("exact_duplicate_identity_only") is not True
        or dedup.get("raw_and_compressed_same_underlying_span_are_duplicates") is not False
        or dedup.get("HCA_and_CSA_overlapping_spans_are_duplicates") is not False
    ):
        raise ValueError("sequence entry identity or deduplication changed")

    union = design.get("one_softmax_union", {})
    if (
        "raw exact window entries plus every" not in union.get("HCA_local_entries", "")
        or "raw exact window entries plus selected" not in union.get("CSA_local_entries", "")
        or "one source-axis softmax denominator" not in union.get("normalization", "")
        or union.get("separately_normalized_branch_sum_forbidden") is not True
        or union.get("raw_span_deduplication_forbidden") is not True
    ):
        raise ValueError("sequence union normalization changed")

    hca_isolation = design.get("HCA_compressor_isolation_v2", {})
    hca_arms = hca_isolation.get("arms", [])
    if (
        [arm.get("id") for arm in hca_arms]
        != [
            "block_mean",
            "static_scalar_position_softmax",
            "static_channel_position_softmax",
            "dynamic_channel_position_softmax",
        ]
        or "exact source compressor" not in hca_arms[-1].get("role", "")
        or len(hca_isolation.get("fixed_across_arms", [])) != 5
        or "cannot hide" not in hca_isolation.get("parameter_accounting", "")
    ):
        raise ValueError("HCA compressor isolation v2 changed")

    csa_contract = design.get("CSA_compressor_and_selector_v2", {})
    if (
        "non-overlapping m-token" not in csa_contract.get("compressor_control", "")
        or "current-a and previous-b" not in csa_contract.get("overlap_intervention", "")
        or csa_contract.get("selection_sequence")
        != [
            "oracle compressed-entry mass",
            "mean compressed index key",
            "learned Lightning-style compressed-entry indexer",
        ]
        or "excluded" not in csa_contract.get("unfinished_current_entry", "")
        or "shared" not in csa_contract.get("head_sharing", "")
    ):
        raise ValueError("CSA compressor or selector v2 changed")

    local = design.get("raw_local_v2", {})
    if (
        "every selected HCA_local or CSA_local slot" not in local.get("placement", "")
        or "one softmax" not in local.get("union", "")
        or "never remove across representation families" not in local.get("deduplication", "")
        or local.get("window_grid") != [0, 64, 128, 256, 512]
        or local.get("published_128_selected_for_Speck") is not False
    ):
        raise ValueError("raw local v2 changed")

    execution = design.get("causal_execution_contract", {})
    sequence = design.get("ordered_experiment_sequence", [])
    if (
        execution.get("source_reference_modes")
        != ["one full prefill at start position zero", "one-token incremental decode"]
        or len(execution.get("source_reference_does_not_cover", [])) != 5
        or len(execution.get("local_successor_must_cover", [])) != 6
        or len(sequence) != 8
        or not sequence[0].startswith("select recurrent backbone")
        or "four HCA compressor arms" not in sequence[1]
        or "raw-local window" not in sequence[3]
        or "oracle compressed-entry selection" not in sequence[5]
        or "all-HCA_local, all-CSA_local" not in sequence[6]
    ):
        raise ValueError("sequence causal or experiment order changed")

    boundary = design.get("registration_and_stop_boundary", {})
    if (
        boundary.get("active_experiment_program_changed") is not False
        or boundary.get("HCA_CSA_raw_local_v1_files_changed") is not False
        or boundary.get("active_architecture_or_model_code_changed") is not False
        or boundary.get("exact_run_configs_materialized") is not False
        or boundary.get("reference_implementation_authorized") is not False
        or boundary.get("training_authorized") is not False
        or boundary.get("promotion_authority") is not False
        or boundary.get("stop_before") != "active finalist acceptance and independent parent selection"
    ):
        raise ValueError("sequence registration or stop boundary changed")
    active = design.get("active_program_boundary", {})
    program_path = root / active.get("program", "")
    if not program_path.is_file() or file_sha256(program_path) != active.get("sha256"):
        raise ValueError("active Paper 1 program changed")
    program = json.loads(program_path.read_text(encoding="utf-8"))
    expected_active = active.get("continues_to_pin", {})
    if (
        program.get("hca_readiness", {}).get("contract") != expected_active.get("HCA")
        or program.get("csa_readiness", {}).get("contract") != expected_active.get("CSA")
        or program.get("raw_local_readiness", {}).get("contract") != expected_active.get("raw_local")
        or active.get("changed") is not False
    ):
        raise ValueError("active sequence-program boundary changed")

    decision = design.get("decision", {})
    false_decisions = (
        "joint_v1_factorization_valid",
        "HCA_compressor_selected",
        "HCA_rate_selected",
        "local_window_selected",
        "CSA_compressor_selected",
        "CSA_selector_selected",
        "HCA_CSA_schedule_selected",
        "parent_selected",
        "implementation_authorized",
        "training_authorized",
        "promotion_authority",
    )
    if decision.get("v2_factorization_corrected") is not True or any(
        decision.get(key) is not False for key in false_decisions
    ):
        raise ValueError("sequence factorization v2 decision changed")
    return {
        "status": "valid_joint_factorization_corrected_training_blocked",
        "layer_types": len(grammar["available_types"]),
        "HCA_compressor_arms": len(hca_arms),
        "experiment_stages": len(sequence),
        "training_authorized": False,
    }


def validate_file(path):
    path = Path(path).resolve()
    return validate_design(json.loads(path.read_text(encoding="utf-8")), path.parents[2])


def main(argv=None):
    report = validate_file(arguments(argv).design)
    print(
        "Sequence compression factorization v2: "
        f"{report['status']} ({report['layer_types']} layer types, "
        f"{report['experiment_stages']} stages)"
    )


if __name__ == "__main__":
    main()
