"""Validate the deconfounded four-arm sequence-cache representation design."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from speck.config import load_experiment
from speck.model import build_model


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


def derive_raw_geometry(root, design):
    source = design["geometry_source"]["candidate_model_config"]
    path = root / source["path"]
    if not path.is_file() or file_sha256(path) != source["sha256"]:
        raise ValueError("cache v2 geometry source changed")
    model = copy.deepcopy(load_experiment(path.parent, "model")["model"])
    attention = []
    ffns = []
    for group in model["blocks"]:
        for stage in group["block"]["stages"]:
            for branch in stage["branches"]:
                if branch["kind"] == "attention":
                    attention.append(branch["num_key_value_heads"])
                    branch["num_key_value_heads"] = 1
                elif branch["kind"] == "swiglu":
                    ffns.append(branch["intermediate_size"])
    if attention != [3] * 5 or ffns != [2304] * 20:
        raise ValueError("cache v2 source is not the frozen five-cache GQA3/FFN2304 backbone")
    model["expected_parameters"] = None
    built = build_model(
        model,
        model["vocab_size"],
        model["bos_token_id"],
        model["eos_token_id"],
    )
    return {
        "parameters": built.parameter_count(),
        "analytic_flops_per_token_at_4096": int(built.flops_per_token(4096)),
        "attention_branches_changed": len(attention),
        "ffn_branches_unchanged": len(ffns),
    }


def validate_design(design, root, geometry=None):
    if (
        design.get("format") != "speck_sequence_cache_representation_design"
        or design.get("format_version") != 2
        or design.get("status") != "corrected_four_arm_design_frozen_training_blocked"
    ):
        raise ValueError("invalid cache representation v2 identity")
    predecessor = design.get("predecessor", {})
    predecessor_path = root / predecessor.get("path", "")
    materialization = design.get("geometry_source", {}).get("materialization", {})
    materialization_path = root / materialization.get("path", "")
    if (
        not predecessor_path.is_file()
        or file_sha256(predecessor_path) != predecessor.get("sha256")
        or not materialization_path.is_file()
        or file_sha256(materialization_path) != materialization.get("sha256")
        or predecessor.get("preserved") is not True
    ):
        raise ValueError("cache representation v2 predecessor or materialization changed")
    geometry = geometry or derive_raw_geometry(root, design)
    arms = {arm.get("id"): arm for arm in design.get("arms", [])}
    if set(arms) != {"gqa3", "mqa1_raw", "mqa1_param_match", "nope_mla128"}:
        raise ValueError("cache representation v2 arms are incomplete")
    control = arms["gqa3"]
    raw = arms["mqa1_raw"]
    matched = arms["mqa1_param_match"]
    if (
        geometry
        != {
            "parameters": 152975898,
            "analytic_flops_per_token_at_4096": 1015703040,
            "attention_branches_changed": 5,
            "ffn_branches_unchanged": 20,
        }
        or control.get("parameters") != 153958938
        or control.get("ffn_intermediate_size") != 2304
        or raw.get("parameters") != geometry["parameters"]
        or raw.get("analytic_flops_per_token_at_4096")
        != geometry["analytic_flops_per_token_at_4096"]
        or raw.get("ffn_intermediate_size") != 2304
        or raw.get("only_model_change")
        != "the five attention num_key_value_heads values change from 3 to 1"
        or matched.get("ffn_intermediate_size") != 2325
        or matched.get("parameters") - raw.get("parameters") != 967680
        or matched.get("analytic_flops_per_token_at_4096")
        - raw.get("analytic_flops_per_token_at_4096")
        != 5806080
        or matched.get("pure_cache_representation_attribution_authorized") is not False
    ):
        raise ValueError("cache representation v2 geometry or attribution changed")
    analysis = design.get("analysis", {})
    stages = design.get("evidence_stages", {})
    registration = design.get("registration_boundary", {})
    correction = design.get("correction", {})
    decision_rule = design.get("decision_rule", {})
    decision = design.get("decision", {})
    if (
        analysis.get("shared_control_candidates") != 3
        or analysis.get("quality_family")
        != [
            "mqa1_raw minus gqa3",
            "mqa1_param_match minus gqa3",
            "nope_mla128 minus gqa3",
        ]
        or not analysis.get("quality_multiplicity", "").startswith("Holm")
        or analysis.get("decomposition_contrast")
        != (
            "mqa1_param_match minus mqa1_raw reported with paired two-sided 95% interval "
            "and no standalone selection authority"
        )
        or analysis.get("no_interim_arm_dropping") is not True
        or stages.get("discovery_model_runs") != 12
        or stages.get("finalist_model_runs") != 24
        or stages.get("training_authorized") is not False
        or registration.get("active_experiment_program_changed") is not False
        or registration.get("v1_future_training_authority") is not False
        or registration.get("v2_training_authority") is not False
        or correction.get("finalist_results_used") is not False
        or correction.get("cache_representation_outputs_present") is not False
        or "representation-level attribution only if mqa1_raw passes"
        not in decision_rule.get("raw_passes", "")
        or "no result may claim MQA1 alone" not in decision_rule.get("raw_fails_matched_passes", "")
        or decision_rule.get("no_quality_cost_trade") is not True
        or decision_rule.get("simpler_cheaper_tie_rule") is not True
        or decision.get("causal_confound_resolved_in_design") is not True
        or decision.get("matched_MQA_pure_representation_attribution") is not False
        or decision.get("training_authorized") is not False
        or decision.get("promotion_authority") is not False
    ):
        raise ValueError("cache representation v2 analysis or decision changed")
    return {
        "status": "valid_training_blocked",
        "arms": len(arms),
        "raw_parameters": geometry["parameters"],
        "raw_flops_per_token_at_4096": geometry["analytic_flops_per_token_at_4096"],
        "training_authorized": False,
    }


def validate_file(path):
    path = Path(path).resolve()
    return validate_design(json.loads(path.read_text(encoding="utf-8")), path.parents[2])


def main(argv=None):
    report = validate_file(arguments(argv).design)
    print(
        "Sequence cache representation v2: "
        f"{report['status']} ({report['arms']} arms, raw={report['raw_parameters']} params)"
    )


if __name__ == "__main__":
    main()
