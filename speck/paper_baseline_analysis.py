"""Collect and analyze preregistered Speck Paper 1 baseline results."""

import hashlib
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

from speck.checkpoint import checkpoint_identity, completed_steps, load_metadata, load_timing
from speck.common import base_dir
from speck.config import load_experiment
from speck.paper_baseline import aligned_steps, load_matrix


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    path = Path(path).expanduser().resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load baseline analysis artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"baseline analysis artifact must contain an object: {path}")
    return path, value


def atomic_json(path, value):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_analysis_plan(path):
    path, plan = load_json(path)
    if (
        plan.get("format") != "speck_paper_baseline_analysis_plan"
        or plan.get("format_version") != 1
        or plan.get("status") != "frozen_before_results"
    ):
        raise ValueError("baseline analysis plan must be frozen format version 1")
    repository_root = path.parents[2]
    matrix_path = repository_root / plan.get("baseline_matrix", "")
    if not matrix_path.is_file() or file_sha256(matrix_path) != plan.get("baseline_matrix_sha256"):
        raise ValueError("baseline analysis matrix does not match its pin")
    _, matrix = load_matrix(matrix_path)
    if plan.get("paper_id") != matrix.get("paper_id") or plan.get("policy_id") != matrix.get(
        "policy_id"
    ):
        raise ValueError("baseline analysis plan does not match its matrix")
    expected_pairs = matrix["planned_primary_baselines"]["proxy_confirmation_pairs"]
    if plan.get("pairs") != expected_pairs:
        raise ValueError("baseline analysis pairs do not match the matrix")
    statistical = plan.get("statistical_contract", {})
    if (
        statistical.get("paired_runs") != len(expected_pairs)
        or statistical.get("candidate_minus_control")
        != "five_cache_kda_gqa-minus-dense_global_param_match"
        or statistical.get("one_sided_confidence_level") != 0.95
        or statistical.get("student_t_critical_df_2") != 2.919985580353724
        or statistical.get("language_loss_non_inferiority_margin_nats") != 0.01
    ):
        raise ValueError("baseline statistical contract is invalid")
    stopping = plan.get("stopping_rule", {})
    if (
        stopping.get("interim_efficacy_looks") != 0
        or stopping.get("interim_futility_looks") != 0
        or stopping.get("required_complete_model_runs") != 2 * len(expected_pairs)
    ):
        raise ValueError("baseline stopping rule is not fixed-sample")
    return path, plan, matrix_path, matrix


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _validate_history(history, expected_steps, final_eval_tokens):
    if not isinstance(history, list) or [entry.get("step") for entry in history] != expected_steps:
        raise ValueError("baseline validation history does not match the frozen cadence")
    previous_optimizer = -1.0
    previous_steady = -1.0
    for index, entry in enumerate(history):
        required = {
            "step",
            "global_step",
            "training_tokens",
            "validation_loss",
            "validation_source_losses",
            "validation_tokens",
            "optimizer_seconds",
            "steady_training_seconds",
        }
        if not required <= set(entry):
            raise ValueError("baseline validation history entry is incomplete")
        numeric = (
            entry["validation_loss"],
            entry["optimizer_seconds"],
            entry["steady_training_seconds"],
        )
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(value) for value in numeric
        ):
            raise ValueError("baseline validation history contains a non-finite value")
        if (
            entry["optimizer_seconds"] < previous_optimizer
            or entry["steady_training_seconds"] < previous_steady
        ):
            raise ValueError("baseline validation timing is not monotonic")
        previous_optimizer = entry["optimizer_seconds"]
        previous_steady = entry["steady_training_seconds"]
        if index == len(history) - 1 and entry["validation_tokens"] != final_eval_tokens:
            raise ValueError("baseline final validation does not use the frozen token count")


def collect_run_result(plan_path, experiment, checkpoint_dir=None):
    """Normalize one completed baseline checkpoint into an auditable result record."""

    plan_path, plan, matrix_path, matrix = load_analysis_plan(plan_path)
    experiment = Path(experiment).expanduser().resolve()
    configs = load_experiment(experiment, "model", "train")
    train = configs["train"]
    planned = matrix["planned_primary_baselines"]
    arm = next((value for value in planned["arms"] if value["id"] == experiment.name), None)
    if arm is None:
        raise ValueError("experiment is not a planned Paper 1 baseline arm")
    pair = next(
        (
            value
            for value in planned["proxy_confirmation_pairs"]
            if value["seed"] == train.get("seed")
            and value["data_token_offset"] == train.get("data_token_offset")
        ),
        None,
    )
    if pair is None:
        raise ValueError("experiment does not match a planned Paper 1 baseline pair")
    pair_id = f"pair-{pair['pair']}-seed-{pair['seed']}-order-{pair['data_token_offset']}"
    expected_run = f"{planned['family_id']}-{pair_id}-{arm['id']}"
    if train.get("run") != expected_run or experiment.parent.name != pair_id:
        raise ValueError("baseline experiment run identity is invalid")
    shared = planned["shared_training"]
    expected_step = aligned_steps(shared["training_tokens"], shared["batch_tokens"])
    checkpoint_dir = (
        Path(checkpoint_dir).expanduser().resolve()
        if checkpoint_dir is not None
        else Path(base_dir()) / "checkpoints" / expected_run
    )
    steps = completed_steps(checkpoint_dir)
    if steps != [expected_step]:
        raise ValueError("baseline run must retain exactly one complete final checkpoint")
    metadata = load_metadata(checkpoint_dir, expected_step)
    resolved = metadata.get("resolved", {})
    expected_resolved = {
        "seed": pair["seed"],
        "data_token_offset": pair["data_token_offset"],
        "train_tokens": shared["training_tokens"],
        "batch_tokens": shared["batch_tokens"],
        "sequence_length": shared["sequence_length"],
        "manifest": shared["data_manifest"],
        "parameters": arm["parameters"],
    }
    actual_resolved = {
        "seed": resolved.get("seed"),
        "data_token_offset": resolved.get("data_token_offset"),
        "train_tokens": resolved.get("train_tokens"),
        "batch_tokens": resolved.get("batch_tokens"),
        "sequence_length": resolved.get("sequence_length"),
        "manifest": metadata.get("manifest"),
        "parameters": resolved.get("parameters"),
    }
    if actual_resolved != expected_resolved:
        differences = sorted(
            key for key in expected_resolved if expected_resolved[key] != actual_resolved[key]
        )
        raise ValueError(f"baseline completed run drifted: {differences}")
    if (
        metadata.get("partial")
        or metadata.get("global_tokens") != shared["training_tokens"]
        or metadata.get("validation_step") != expected_step
        or metadata.get("validation_tokens") != shared["final_evaluation_tokens"]
    ):
        raise ValueError("baseline final checkpoint is incomplete")
    cadence = shared["evaluation_every_steps"]
    expected_validation_steps = [0, cadence, 2 * cadence, 3 * cadence, 4 * cadence, expected_step]
    history = metadata.get("validation_history")
    _validate_history(history, expected_validation_steps, shared["final_evaluation_tokens"])
    summary_path = checkpoint_dir / "run_summary.json"
    _, summary = load_json(summary_path)
    if (
        summary.get("partial")
        or summary.get("completed_steps") != expected_step
        or summary.get("global_tokens") != shared["training_tokens"]
        or summary.get("validation_history") != history
    ):
        raise ValueError("baseline run summary does not match its final checkpoint")
    timing = load_timing(checkpoint_dir, expected_step)
    if timing is None:
        raise ValueError("baseline run is missing final timing evidence")
    final = history[-1]
    complete_path = checkpoint_dir / f"complete_{expected_step:06d}"
    return {
        "format": "speck_paper_baseline_run_result",
        "format_version": 1,
        "status": "complete_qualified",
        "created_at": _utc_now(),
        "checkpoint_completed_at": datetime.fromtimestamp(
            complete_path.stat().st_mtime, timezone.utc
        ).isoformat(),
        "paper_id": plan["paper_id"],
        "policy_id": plan["policy_id"],
        "analysis_plan_sha256": file_sha256(plan_path),
        "baseline_matrix_sha256": file_sha256(matrix_path),
        "pair": pair,
        "arm_id": arm["id"],
        "experiment": str(experiment),
        "run": expected_run,
        "checkpoint": checkpoint_identity(checkpoint_dir, expected_step),
        "run_summary": {
            "path": str(summary_path),
            "sha256": file_sha256(summary_path),
        },
        "parameters": arm["parameters"],
        "flops_per_token_at_4096": arm["flops_per_token_at_4096"],
        "training_tokens": shared["training_tokens"],
        "validation_history": history,
        "final_validation": final,
        "timing": timing,
        "peak_allocated_bytes": metadata.get("peak_allocated_bytes"),
        "non_finite_steps": 0,
    }


def _validate_result(report, plan, matrix, plan_sha256, matrix_sha256):
    required = {
        "format",
        "format_version",
        "status",
        "created_at",
        "paper_id",
        "policy_id",
        "analysis_plan_sha256",
        "baseline_matrix_sha256",
        "pair",
        "arm_id",
        "parameters",
        "flops_per_token_at_4096",
        "training_tokens",
        "validation_history",
        "final_validation",
        "non_finite_steps",
    }
    if not required <= set(report):
        raise ValueError("baseline run result is incomplete")
    if (
        report["format"] != "speck_paper_baseline_run_result"
        or report["format_version"] != 1
        or report["status"] != "complete_qualified"
        or report["paper_id"] != plan["paper_id"]
        or report["policy_id"] != plan["policy_id"]
        or report["analysis_plan_sha256"] != plan_sha256
        or report["baseline_matrix_sha256"] != matrix_sha256
        or report["non_finite_steps"] != 0
    ):
        raise ValueError("baseline run result identity or completion status is invalid")
    if report["pair"] not in plan["pairs"] or report["arm_id"] not in plan["arms"].values():
        raise ValueError("baseline run result is outside the frozen design")
    planned = matrix["planned_primary_baselines"]
    arm = next(value for value in planned["arms"] if value["id"] == report["arm_id"])
    shared = planned["shared_training"]
    if (
        report["parameters"] != arm["parameters"]
        or report["flops_per_token_at_4096"] != arm["flops_per_token_at_4096"]
        or report["training_tokens"] != shared["training_tokens"]
    ):
        raise ValueError("baseline run result geometry is invalid")
    _validate_history(
        report["validation_history"],
        plan["input_contract"]["validation_steps"],
        plan["input_contract"]["final_validation_tokens"],
    )
    if report["final_validation"] != report["validation_history"][-1]:
        raise ValueError("baseline final validation is not the last trace point")
    for entry in report["validation_history"]:
        if not math.isfinite(entry["validation_loss"]):
            raise ValueError("baseline run result contains non-finite loss")


def _load_results(plan_path, result_paths):
    plan_path, plan, matrix_path, matrix = load_analysis_plan(plan_path)
    plan_sha256 = file_sha256(plan_path)
    matrix_sha256 = file_sha256(matrix_path)
    results = []
    references = []
    for path in result_paths:
        path, report = load_json(path)
        _validate_result(report, plan, matrix, plan_sha256, matrix_sha256)
        results.append(report)
        references.append({"path": str(path), "sha256": file_sha256(path)})
    return plan_path, plan, matrix_path, matrix, results, references


def lock_time_to_quality_target(plan_path, control_result_paths):
    """Lock the time-to-quality target using only completed dense-control results."""

    plan_path, plan, matrix_path, _, results, references = _load_results(
        plan_path, control_result_paths
    )
    control = plan["arms"]["control"]
    if len(results) != len(plan["pairs"]) or any(result["arm_id"] != control for result in results):
        raise ValueError("time-to-quality locking requires exactly the three control results")
    if {result["pair"]["pair"] for result in results} != {pair["pair"] for pair in plan["pairs"]}:
        raise ValueError("time-to-quality controls do not cover every frozen pair")
    decimals = plan["time_to_quality_target"]["round_up_decimals"]
    scale = 10**decimals
    maximum = max(result["final_validation"]["validation_loss"] for result in results)
    target = math.ceil(maximum * scale) / scale
    return {
        "format": "speck_paper_baseline_time_to_quality_lock",
        "format_version": 1,
        "status": "locked_from_controls_before_candidate_analysis",
        "locked_at": _utc_now(),
        "paper_id": plan["paper_id"],
        "policy_id": plan["policy_id"],
        "analysis_plan_sha256": file_sha256(plan_path),
        "baseline_matrix_sha256": file_sha256(matrix_path),
        "control_arm": control,
        "control_results": references,
        "rule": plan["time_to_quality_target"]["rule"],
        "round_up_decimals": decimals,
        "validation_loss_target": target,
    }


def _interpolate(trace, field, target):
    points = [(entry[field], entry["validation_loss"]) for entry in trace]
    if target < points[0][0] or target > points[-1][0]:
        raise ValueError(f"interpolation target is outside the {field} trace")
    for (left_x, left_y), (right_x, right_y) in zip(points, points[1:]):
        if target == left_x:
            return left_y
        if left_x <= target <= right_x:
            if right_x == left_x:
                return right_y
            weight = (target - left_x) / (right_x - left_x)
            return left_y + weight * (right_y - left_y)
    return points[-1][1]


def _first_crossing(trace, target):
    if trace[0]["validation_loss"] <= target:
        return trace[0]["steady_training_seconds"]
    for left, right in zip(trace, trace[1:]):
        left_loss = left["validation_loss"]
        right_loss = right["validation_loss"]
        if right_loss > target:
            continue
        if right_loss == left_loss:
            return right["steady_training_seconds"]
        weight = (left_loss - target) / (left_loss - right_loss)
        return left["steady_training_seconds"] + weight * (
            right["steady_training_seconds"] - left["steady_training_seconds"]
        )
    return None


def _paired_summary(values, t_critical):
    values = list(values)
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    standard_error = standard_deviation / math.sqrt(len(values))
    return {
        "n": len(values),
        "values": values,
        "mean": mean,
        "sample_standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "upper_one_sided_95_bound": mean + t_critical * standard_error,
        "lower_one_sided_95_bound": mean - t_critical * standard_error,
    }


def analyze_baselines(plan_path, target_lock_path, result_paths):
    """Apply the frozen fixed-sample analysis to all six baseline results."""

    plan_path, plan, matrix_path, matrix, results, references = _load_results(
        plan_path, result_paths
    )
    target_lock_path, target_lock = load_json(target_lock_path)
    if (
        target_lock.get("format") != "speck_paper_baseline_time_to_quality_lock"
        or target_lock.get("status") != "locked_from_controls_before_candidate_analysis"
        or target_lock.get("analysis_plan_sha256") != file_sha256(plan_path)
        or target_lock.get("baseline_matrix_sha256") != file_sha256(matrix_path)
    ):
        raise ValueError("time-to-quality target lock is invalid")
    arms = plan["arms"]
    expected_cells = {(pair["pair"], arm) for pair in plan["pairs"] for arm in arms.values()}
    observed_cells = {(result["pair"]["pair"], result["arm_id"]) for result in results}
    if len(results) != len(expected_cells) or observed_cells != expected_cells:
        raise ValueError("baseline analysis requires exactly one result for every frozen cell")
    candidate_reports = [result for result in results if result["arm_id"] == arms["candidate"]]
    locked_at = datetime.fromisoformat(target_lock["locked_at"].replace("Z", "+00:00"))
    if any(
        datetime.fromisoformat(result["created_at"].replace("Z", "+00:00")) <= locked_at
        for result in candidate_reports
    ):
        raise ValueError("candidate result records must be created after target locking")
    by_cell = {(result["pair"]["pair"], result["arm_id"]): result for result in results}
    t_critical = plan["statistical_contract"]["student_t_critical_df_2"]
    margin = plan["statistical_contract"]["language_loss_non_inferiority_margin_nats"]
    compute_tokens = matrix["planned_primary_baselines"]["matching_views"]["compute_matched"]
    paired = []
    token_differences = []
    compute_differences = []
    wall_clock_differences = []
    time_improvements = []
    censored_pairs = []
    target = target_lock["validation_loss_target"]
    for pair in plan["pairs"]:
        control = by_cell[(pair["pair"], arms["control"])]
        candidate = by_cell[(pair["pair"], arms["candidate"])]
        token_difference = (
            candidate["final_validation"]["validation_loss"]
            - control["final_validation"]["validation_loss"]
        )
        compute_difference = _interpolate(
            candidate["validation_history"],
            "training_tokens",
            compute_tokens["reference_tokens"],
        ) - _interpolate(
            control["validation_history"],
            "training_tokens",
            compute_tokens["dense_global_tokens"],
        )
        wall_budget = min(
            control["validation_history"][-1]["steady_training_seconds"],
            candidate["validation_history"][-1]["steady_training_seconds"],
        )
        wall_difference = _interpolate(
            candidate["validation_history"], "steady_training_seconds", wall_budget
        ) - _interpolate(control["validation_history"], "steady_training_seconds", wall_budget)
        control_time = _first_crossing(control["validation_history"], target)
        candidate_time = _first_crossing(candidate["validation_history"], target)
        time_result = {
            "control_seconds": control_time,
            "candidate_seconds": candidate_time,
            "right_censored": control_time is None or candidate_time is None,
        }
        if time_result["right_censored"]:
            censored_pairs.append(pair["pair"])
        else:
            improvement = 1 - candidate_time / control_time if control_time else 0.0
            time_result["candidate_relative_improvement"] = improvement
            time_improvements.append(improvement)
        paired.append(
            {
                "pair": pair,
                "fixed_tokens_candidate_minus_control_loss": token_difference,
                "fixed_flops_candidate_minus_control_loss": compute_difference,
                "fixed_steady_time_seconds": wall_budget,
                "fixed_steady_time_candidate_minus_control_loss": wall_difference,
                "time_to_quality": time_result,
            }
        )
        token_differences.append(token_difference)
        compute_differences.append(compute_difference)
        wall_clock_differences.append(wall_difference)

    source_names = set.intersection(
        *(set(result["final_validation"]["validation_source_losses"]) for result in results)
    )
    if not source_names or any(
        set(result["final_validation"]["validation_source_losses"]) != source_names
        for result in results
    ):
        raise ValueError("baseline source-loss guardrails require identical non-empty sources")
    source_guardrails = {}
    source_margin = plan["statistical_contract"]["source_guardrail_nats"]
    for source in sorted(source_names):
        differences = []
        for pair in plan["pairs"]:
            control = by_cell[(pair["pair"], arms["control"])]
            candidate = by_cell[(pair["pair"], arms["candidate"])]
            differences.append(
                candidate["final_validation"]["validation_source_losses"][source]
                - control["final_validation"]["validation_source_losses"][source]
            )
        summary = _paired_summary(differences, t_critical)
        summary["margin_nats"] = source_margin
        summary["pass"] = summary["upper_one_sided_95_bound"] <= source_margin
        source_guardrails[source] = summary

    fixed_tokens = _paired_summary(token_differences, t_critical)
    fixed_tokens["margin_nats"] = margin
    fixed_tokens["non_inferiority_pass"] = fixed_tokens["upper_one_sided_95_bound"] <= margin
    all_sources_pass = all(value["pass"] for value in source_guardrails.values())
    return {
        "format": "speck_paper_baseline_analysis",
        "format_version": 1,
        "status": "complete_proxy_evidence_no_promotion_authority",
        "created_at": _utc_now(),
        "paper_id": plan["paper_id"],
        "policy_id": plan["policy_id"],
        "analysis_plan_sha256": file_sha256(plan_path),
        "baseline_matrix_sha256": file_sha256(matrix_path),
        "time_to_quality_lock": {
            "path": str(target_lock_path),
            "sha256": file_sha256(target_lock_path),
            "validation_loss_target": target,
        },
        "run_results": references,
        "paired_results": paired,
        "fixed_tokens": fixed_tokens,
        "fixed_analytic_flops": _paired_summary(compute_differences, t_critical),
        "fixed_steady_training_time": _paired_summary(wall_clock_differences, t_critical),
        "time_to_quality": {
            "right_censored_pairs": censored_pairs,
            "paired_relative_improvement": (
                _paired_summary(time_improvements, t_critical) if not censored_pairs else None
            ),
        },
        "source_guardrails": source_guardrails,
        "proxy_quality_screen_pass": fixed_tokens["non_inferiority_pass"] and all_sources_pass,
        "authority": plan["decision_authority"],
    }
