"""Validate the pre-result GDN/KDA by RoPE/NoPE factorial design."""

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


def differences(left, right, path=""):
    if type(left) is not type(right):
        return [(path, left, right)]
    if isinstance(left, dict):
        result = []
        for key in sorted(set(left) | set(right)):
            result.extend(differences(left.get(key), right.get(key), f"{path}/{key}"))
        return result
    if isinstance(left, list):
        if len(left) != len(right):
            return [(path, left, right)]
        return [
            item
            for index, values in enumerate(zip(left, right))
            for item in differences(*values, f"{path}/{index}")
        ]
    return [] if left == right else [(path, left, right)]


def derive_config_evidence(root, design):
    configs = design["config_evidence"]
    loaded = {}
    for name in ("GDN_RoPE", "GDN_NoPE", "KDA_NoPE"):
        reference = configs[name]
        path = root / reference["path"]
        if not path.is_file() or file_sha256(path) != reference["sha256"]:
            raise ValueError("NoPE factorial config evidence changed")
        loaded[name] = load_experiment(path.parent, "model")["model"]
    observed = differences(loaded["GDN_RoPE"], loaded["GDN_NoPE"])
    kda_rope = copy.deepcopy(loaded["KDA_NoPE"])
    changed = 0
    for group in kda_rope["blocks"]:
        for stage in group["block"]["stages"]:
            for branch in stage["branches"]:
                if branch["kind"] == "attention":
                    branch["rope_dim"] = 32
                    changed += 1
    kda_rope["expected_parameters"] = None
    built = build_model(
        kda_rope,
        kda_rope["vocab_size"],
        kda_rope["bos_token_id"],
        kda_rope["eos_token_id"],
    )
    return {
        "GDN_differences": observed,
        "KDA_attention_branches_changed": changed,
        "KDA_RoPE_parameters": built.parameter_count(),
        "KDA_RoPE_flops": int(built.flops_per_token(4096)),
    }


def validate_design(design, root, derived=None):
    if (
        design.get("format") != "speck_nope_factorial_design"
        or design.get("format_version") != 1
        or design.get("status")
        != "factorial_frozen_before_KDA_NoPE_finalist_outputs_training_blocked"
    ):
        raise ValueError("invalid NoPE factorial identity")
    for value in design.get("historical_evidence", {}).values():
        path_value = value.get("result") or value.get("finding")
        if path_value and (
            not (root / path_value).is_file()
            or file_sha256(root / path_value) != value.get("sha256")
        ):
            raise ValueError("NoPE factorial historical evidence changed")
    historical = design["historical_evidence"]["seed42_staircase"]
    summary = json.loads((root / historical["result"]).read_text(encoding="utf-8"))
    if (
        historical.get("GDN_RoPE_loss")
        != summary["runs"]["gdn-fla-sigmoid-rope"]["final_validation_loss"]
        or historical.get("GDN_NoPE_loss")
        != summary["runs"]["gdn-fla-sigmoid-nope"]["final_validation_loss"]
        or historical.get("NoPE_minus_RoPE_loss")
        != summary["runs"]["gdn-fla-sigmoid-nope"]["delta_from_previous"]
        or historical.get("same_mixer_same_parameters_same_flops") is not True
    ):
        raise ValueError("NoPE factorial seed-42 contrast changed")
    cells = design.get("confirmation_cells", {})
    cell_path = root / cells.get("source", "")
    reuse = design.get("reuse_contract", {})
    reuse_qualification = reuse.get("v3_acceptance_qualification", {})
    active_materialization = reuse.get("active_materialization", {})
    active_arm_model = reuse.get("active_arm_model", {})
    if (
        not cell_path.is_file()
        or file_sha256(cell_path) != cells.get("sha256")
        or not (root / reuse_qualification.get("path", "")).is_file()
        or file_sha256(root / reuse_qualification["path"]) != reuse_qualification.get("sha256")
        or not (root / active_materialization.get("path", "")).is_file()
        or file_sha256(root / active_materialization["path"])
        != active_materialization.get("sha256")
        or not (root / active_arm_model.get("path", "")).is_file()
        or file_sha256(root / active_arm_model["path"]) != active_arm_model.get("sha256")
    ):
        raise ValueError("NoPE factorial cells or reuse qualification changed")
    derived = derived or derive_config_evidence(root, design)
    expected_paths = [
        f"/blocks/{index}/block/stages/0/branches/0/rope_dim" for index in (3, 7, 11, 15, 19)
    ]
    clean_delta = design["config_evidence"]["clean_GDN_delta"]
    if (
        [value[0] for value in derived["GDN_differences"]] != expected_paths
        or any((left, right) != (32, 0) for _, left, right in derived["GDN_differences"])
        or clean_delta.get("differences") != 5
        or [f"/{path}" for path in clean_delta.get("paths", ())] != expected_paths
        or clean_delta.get("control_value") != 32
        or clean_delta.get("treatment_value") != 0
        or derived["KDA_attention_branches_changed"] != 5
        or derived["KDA_RoPE_parameters"] != 153958938
        or derived["KDA_RoPE_flops"] != 1021601280
    ):
        raise ValueError("NoPE factorial config isolation or KDA/RoPE geometry changed")
    arms = {arm.get("id"): arm for arm in design.get("factorial", {}).get("arms", [])}
    if set(arms) != {
        "gdn_sigmoid_rope32",
        "gdn_sigmoid_nope0",
        "kda_sigmoid_rope32",
        "kda_sigmoid_nope0",
    }:
        raise ValueError("NoPE factorial arm inventory changed")
    if (
        arms["gdn_sigmoid_rope32"]["parameters"] != arms["gdn_sigmoid_nope0"]["parameters"]
        or arms["kda_sigmoid_rope32"]["parameters"] != arms["kda_sigmoid_nope0"]["parameters"]
        or arms["kda_sigmoid_rope32"]["parameters"] != 153958938
    ):
        raise ValueError("NoPE factorial within-mixer geometry changed")
    plan = json.loads(cell_path.read_text(encoding="utf-8"))
    expected_pairs = [
        {key: pair[key] for key in ("pair", "seed", "data_token_offset")} for pair in plan["pairs"]
    ]
    factorial = design["factorial"]
    statistical = design.get("statistical_contract", {})
    decision = design.get("decision_rule", {})
    registration = design.get("registration_boundary", {})
    if (
        cells.get("pairs") != expected_pairs
        or len(cells.get("pairs", ())) != 6
        or factorial.get("within_mixer_position_contrasts")
        != [
            "gdn_sigmoid_nope0 minus gdn_sigmoid_rope32",
            "kda_sigmoid_nope0 minus kda_sigmoid_rope32",
        ]
        or factorial.get("interaction")
        != "(kda_sigmoid_nope0-kda_sigmoid_rope32)-(gdn_sigmoid_nope0-gdn_sigmoid_rope32)"
        or statistical.get("fixed_data_order_strata") != 2
        or statistical.get("independent_seeds_per_stratum") != 3
        or statistical.get("student_t_critical_df_2_one_sided_95") != 2.919985580353724
        or not statistical.get("position_contrast_multiplicity", "").startswith("Holm")
        or statistical.get("all_four_arms_required") is not True
        or statistical.get("interim_arm_dropping") is not False
        or reuse.get("accepted_results_required") != 6
        or reuse.get("current_accepted_KDA_NoPE_results") != 0
        or reuse.get("outcomes_inspected") is not False
        or reuse.get("new_future_runs") != 18
        or reuse.get("total_factorial_runs") != 24
        or decision.get("no_general_NoPE_claim_from_finalist") is not True
        or decision.get("no_quality_capability_tradeoff") is not True
        or registration.get("active_experiment_program_changed") is not False
        or registration.get("active_finalist_execution_changed") is not False
        or registration.get("training_authorized") is not False
        or registration.get("promotion_authority") is not False
    ):
        raise ValueError("NoPE factorial analysis, reuse, or decision changed")
    return {
        "status": "valid_training_blocked",
        "arms": len(arms),
        "pairs": len(expected_pairs),
        "reused_future_runs": reuse["accepted_results_required"],
        "new_future_runs": reuse["new_future_runs"],
        "training_authorized": False,
    }


def validate_file(path):
    path = Path(path).resolve()
    return validate_design(json.loads(path.read_text(encoding="utf-8")), path.parents[2])


def main(argv=None):
    report = validate_file(arguments(argv).design)
    print(
        "NoPE factorial: "
        f"{report['status']} ({report['arms']} arms, {report['pairs']} pairs, "
        f"{report['new_future_runs']} new runs)"
    )


if __name__ == "__main__":
    main()
