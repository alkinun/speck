"""Validate raw and parameter-matched ratio geometry and causal roles."""

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


def derive_geometries(root, design):
    source = design["geometry_source"]["candidate_model_config"]
    path = root / source["path"]
    if not path.is_file() or file_sha256(path) != source["sha256"]:
        raise ValueError("ratio v2 geometry source changed")
    model = load_experiment(path.parent, "model")["model"]
    layers = []
    for group in model["blocks"]:
        for _ in range(group.get("repeat", 1)):
            value = copy.deepcopy(group)
            value["repeat"] = 1
            layers.append(value)
    if len(layers) != 20:
        raise ValueError("ratio v2 source does not contain twenty layers")
    attention = next(
        copy.deepcopy(value)
        for value in layers
        if value["block"]["stages"][0]["branches"][0]["kind"] == "attention"
    )
    recurrent = next(
        copy.deepcopy(value)
        for value in layers
        if value["block"]["stages"][0]["branches"][0]["kind"] == "kimi_delta_attention"
    )
    specifications = {
        "ratio_1_to_1_raw": ([1, 3, 5, 7, 9, 11, 13, 15, 17, 19], 2304),
        "ratio_3_to_1_raw": ([3, 7, 11, 15, 19], 2304),
        "ratio_9_to_1_raw": ([9, 19], 2304),
        "ratio_1_to_1_param_match": ([1, 3, 5, 7, 9, 11, 13, 15, 17, 19], 2281),
        "ratio_9_to_1_param_match": ([9, 19], 2318),
    }
    geometries = {}
    for name, (indices, width) in specifications.items():
        settings = copy.deepcopy(model)
        settings["blocks"] = [
            copy.deepcopy(attention if index in indices else recurrent) for index in range(20)
        ]
        for group in settings["blocks"]:
            for stage in group["block"]["stages"]:
                for branch in stage["branches"]:
                    if branch["kind"] == "swiglu":
                        branch["intermediate_size"] = width
        settings["expected_parameters"] = None
        built = build_model(
            settings,
            settings["vocab_size"],
            settings["bos_token_id"],
            settings["eos_token_id"],
        )
        geometries[name] = {
            "parameters": built.parameter_count(),
            "analytic_flops_per_token_at_4096": int(built.flops_per_token(4096)),
            "global_layers": len(indices),
            "recurrent_layers": 20 - len(indices),
            "FFN_intermediate_size": width,
        }
    return geometries


EXPECTED = {
    "ratio_1_to_1_raw": {
        "parameters": 155024828,
        "analytic_flops_per_token_at_4096": 1121172480,
        "global_layers": 10,
        "recurrent_layers": 10,
        "FFN_intermediate_size": 2304,
    },
    "ratio_3_to_1_raw": {
        "parameters": 153958938,
        "analytic_flops_per_token_at_4096": 1021601280,
        "global_layers": 5,
        "recurrent_layers": 15,
        "FFN_intermediate_size": 2304,
    },
    "ratio_9_to_1_raw": {
        "parameters": 153319404,
        "analytic_flops_per_token_at_4096": 961858560,
        "global_layers": 2,
        "recurrent_layers": 18,
        "FFN_intermediate_size": 2304,
    },
    "ratio_1_to_1_param_match": {
        "parameters": 153964988,
        "analytic_flops_per_token_at_4096": 1114813440,
        "global_layers": 10,
        "recurrent_layers": 10,
        "FFN_intermediate_size": 2281,
    },
    "ratio_9_to_1_param_match": {
        "parameters": 153964524,
        "analytic_flops_per_token_at_4096": 965729280,
        "global_layers": 2,
        "recurrent_layers": 18,
        "FFN_intermediate_size": 2318,
    },
}


def validate_design(design, root, geometries=None):
    if (
        design.get("format") != "speck_ratio_placement_readiness_gate"
        or design.get("format_version") != 2
        or design.get("status") != "deconfounded_raw_ratio_design_frozen_training_blocked"
    ):
        raise ValueError("invalid ratio placement v2 identity")
    predecessor = design.get("predecessor", {})
    materialization = design.get("geometry_source", {}).get("materialization", {})
    if (
        not (root / predecessor.get("path", "")).is_file()
        or file_sha256(root / predecessor["path"]) != predecessor.get("sha256")
        or not (root / materialization.get("path", "")).is_file()
        or file_sha256(root / materialization["path"]) != materialization.get("sha256")
        or predecessor.get("preserved") is not True
    ):
        raise ValueError("ratio placement v2 predecessor or materialization changed")
    geometries = geometries or derive_geometries(root, design)
    if geometries != EXPECTED:
        raise ValueError("ratio placement v2 code-derived geometry changed")
    raw = {value["id"]: value for value in design.get("raw_selection_arms", [])}
    matched = {value["id"]: value for value in design.get("parameter_matched_sensitivities", [])}
    if set(raw) != set(list(EXPECTED)[:3]) or set(matched) != set(list(EXPECTED)[3:]):
        raise ValueError("ratio placement v2 arm inventory changed")
    for name, geometry in geometries.items():
        record = raw.get(name) or matched.get(name)
        if any(record.get(key) != value for key, value in geometry.items()):
            raise ValueError("ratio placement v2 recorded geometry changed")
    if any(
        record.get("pure_ratio_attribution_authorized") is not False
        or record.get("promotion_authority") is not False
        for record in matched.values()
    ):
        raise ValueError("ratio placement v2 matched sensitivity gained authority")
    contrasts = design.get("causal_contrasts", {})
    stages = design.get("stages", {})
    selection = design.get("selection", {})
    placement = design.get("placement_successor", {})
    registration = design.get("registration_boundary", {})
    decision = design.get("decision", {})
    if (
        contrasts.get("raw_selection_family")
        != [
            "ratio_1_to_1_raw minus ratio_3_to_1_raw",
            "ratio_9_to_1_raw minus ratio_3_to_1_raw",
        ]
        or len(contrasts.get("capacity_sensitivities", ())) != 2
        or contrasts.get("matched_compound_pure_ratio_attribution") is not False
        or stages.get("discovery", {}).get("model_runs") != 15
        or stages.get("finalist", {}).get("model_runs") != 18
        or stages.get("finalist", {}).get("matched_sensitivity_repeated_automatically") is not False
        or selection.get("eligible_arms") != list(EXPECTED)[:3]
        or selection.get("matched_sensitivity_can_select") is not False
        or selection.get("no_quality_cost_trade") is not True
        or placement.get("status") != "not_frozen_until_one_raw_ratio_is_selected"
        or placement.get("matched_ratio_sensitivity_cannot_choose_placement_count") is not True
        or registration.get("active_experiment_program_changed") is not False
        or registration.get("v1_future_ratio_training_authorized") is not False
        or registration.get("v2_training_authorized") is not False
        or decision.get("ratio_FFN_confound_resolved_in_design") is not True
        or decision.get("training_authorized") is not False
        or decision.get("promotion_authority") is not False
    ):
        raise ValueError("ratio placement v2 analysis or decision changed")
    return {
        "status": "valid_training_blocked",
        "raw_arms": len(raw),
        "matched_sensitivities": len(matched),
        "discovery_runs": stages["discovery"]["model_runs"],
        "training_authorized": False,
    }


def validate_file(path):
    path = Path(path).resolve()
    return validate_design(json.loads(path.read_text(encoding="utf-8")), path.parents[2])


def main(argv=None):
    report = validate_file(arguments(argv).design)
    print(
        "Ratio placement v2: "
        f"{report['status']} ({report['raw_arms']} raw, "
        f"{report['matched_sensitivities']} sensitivities)"
    )


if __name__ == "__main__":
    main()
