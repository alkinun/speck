"""Validate the conditional recurrent SiLU/sigmoid readiness design."""

import argparse
import dataclasses
import hashlib
import inspect
import json
from pathlib import Path

from speck.architecture import KimiDeltaAttentionSpec
from speck.config import load_experiment
from speck.model import KimiDeltaAttention


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


def derive_evidence(root, design):
    configs = design["config_evidence"]
    loaded = {}
    for name in ("GDN_SiLU_RoPE", "GDN_sigmoid_RoPE"):
        reference = configs[name]
        path = root / reference["path"]
        if not path.is_file() or file_sha256(path) != reference["sha256"]:
            raise ValueError("recurrent gate config evidence changed")
        loaded[name] = load_experiment(path.parent, "model")["model"]
    fields = {field.name for field in dataclasses.fields(KimiDeltaAttentionSpec)}
    source = inspect.getsource(KimiDeltaAttention.forward)
    return {
        "GDN_differences": differences(loaded["GDN_SiLU_RoPE"], loaded["GDN_sigmoid_RoPE"]),
        "KDA_activation_field_present": "output_gate_activation" in fields,
        "KDA_forward_hardcoded_sigmoid": "output_gate.float().sigmoid()" in source,
    }


def validate_design(design, root, derived=None):
    if (
        design.get("format") != "speck_recurrent_output_gate_readiness"
        or design.get("format_version") != 1
        or design.get("status")
        != "conditional_design_frozen_KDA_SiLU_unimplemented_training_blocked"
    ):
        raise ValueError("invalid recurrent gate readiness identity")
    for value in design.get("historical_evidence", {}).values():
        path_value = value.get("result") or value.get("finding")
        if path_value and (
            not (root / path_value).is_file()
            or file_sha256(root / path_value) != value.get("sha256")
        ):
            raise ValueError("recurrent gate historical evidence changed")
    summary_record = design["historical_evidence"]["seed42_GDN_language"]
    summary = json.loads((root / summary_record["result"]).read_text(encoding="utf-8"))
    if (
        summary_record.get("SiLU_loss")
        != summary["runs"]["gdn-fla-silu-rope"]["final_validation_loss"]
        or summary_record.get("sigmoid_loss")
        != summary["runs"]["gdn-fla-sigmoid-rope"]["final_validation_loss"]
        or summary_record.get("sigmoid_minus_SiLU")
        != summary["runs"]["gdn-fla-sigmoid-rope"]["delta_from_previous"]
    ):
        raise ValueError("recurrent gate historical language contrast changed")
    implementation = design.get("current_KDA_implementation", {})
    synthetic = design["historical_evidence"]["synthetic_MQAR"]
    for reference in (
        implementation.get("architecture_module", {}),
        implementation.get("model_module", {}),
    ):
        path = root / reference.get("path", "")
        if not path.is_file() or file_sha256(path) != reference.get("sha256"):
            raise ValueError("recurrent gate implementation source changed")
    parent = design.get("ordered_prerequisites", {}).get("parent_selection", {})
    for key, hash_key in (("design", "sha256"), ("qualification", "qualification_sha256")):
        path = root / parent.get(key, "")
        if not path.is_file() or file_sha256(path) != parent.get(hash_key):
            raise ValueError("recurrent gate parent-selection evidence changed")
    derived = derived or derive_evidence(root, design)
    recurrent_indices = (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18)
    expected_paths = [
        f"/blocks/{index}/block/stages/0/branches/0/output_gate_activation"
        for index in recurrent_indices
    ]
    if (
        [value[0] for value in derived["GDN_differences"]] != expected_paths
        or any(
            (left, right) != ("silu", "sigmoid") for _, left, right in derived["GDN_differences"]
        )
        or derived["KDA_activation_field_present"] is not False
        or derived["KDA_forward_hardcoded_sigmoid"] is not True
        or implementation.get("KimiDeltaAttentionSpec_output_gate_activation_field") is not False
        or implementation.get("KDA_SiLU_config_expressible") is not False
        or implementation.get("attempted_extra_field_result")
        != "TypeError unexpected keyword argument output_gate_activation"
        or synthetic.get("GDN_SiLU_pass_seeds") != 3
        or synthetic.get("GDN_sigmoid_pass_seeds") != 0
        or synthetic.get("KDA_sigmoid_pass_seeds") != 3
        or synthetic.get("KDA_SiLU_cell_present") is not False
    ):
        raise ValueError("recurrent gate clean delta or KDA implementation boundary changed")
    experiment = design.get("conditional_experiment", {})
    analysis = design.get("analysis", {})
    decision = design.get("decision_rule", {})
    registration = design.get("registration_boundary", {})
    plan_path = root / experiment.get("paired_cells_source", "")
    if (
        not plan_path.is_file()
        or file_sha256(plan_path) != experiment.get("paired_cells_sha256")
        or experiment.get("parent")
        != (
            "exact selected mixer, global position treatment, layer count/placement, cache "
            "representation, FFN, residual, tokenizer, data, optimizer, and evaluation"
        )
        or [arm.get("output_gate_activation") for arm in experiment.get("arms", [])]
        != ["silu", "sigmoid"]
        or experiment.get("only_model_difference")
        != "output_gate_activation on every recurrent branch"
        or experiment.get("pairs") != 6
        or experiment.get("new_SiLU_runs_if_sigmoid_reused") != 6
        or experiment.get("new_runs_if_no_exact_reuse") != 12
        or experiment.get("no_interim_arm_dropping") is not True
        or analysis.get("fixed_data_order_strata") != 2
        or analysis.get("independent_seeds_per_stratum") != 3
        or analysis.get("one_sided_t_critical_df_2") != 2.919985580353724
        or decision.get("conditional_only") is not True
        or decision.get("general_sigmoid_claim") is not False
        or decision.get("synthetic_or_language_result_can_override_other_hard_gate") is not False
        or registration.get("active_experiment_program_changed") is not False
        or registration.get("active_architecture_or_model_code_changed") is not False
        or registration.get("KDA_gate_implementation_authorized") is not False
        or registration.get("training_authorized") is not False
        or registration.get("promotion_authority") is not False
    ):
        raise ValueError("recurrent gate experiment, analysis, or decision changed")
    return {
        "status": "valid_training_blocked",
        "clean_GDN_gate_paths": len(expected_paths),
        "KDA_SiLU_expressible": False,
        "conditional_pairs": experiment["pairs"],
        "training_authorized": False,
    }


def validate_file(path):
    path = Path(path).resolve()
    return validate_design(json.loads(path.read_text(encoding="utf-8")), path.parents[2])


def main(argv=None):
    report = validate_file(arguments(argv).design)
    print(
        "Recurrent output gate readiness: "
        f"{report['status']} ({report['clean_GDN_gate_paths']} GDN paths, "
        f"KDA-SiLU={str(report['KDA_SiLU_expressible']).lower()})"
    )


if __name__ == "__main__":
    main()
