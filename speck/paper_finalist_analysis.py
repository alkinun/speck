"""Collect and analyze the frozen Paper 1 six-pair finalist evidence."""

import math
from datetime import datetime, timezone
from pathlib import Path

from speck.checkpoint import checkpoint_identity, completed_steps, load_metadata, load_timing
from speck.config import load_experiment
from speck.paper_baseline_analysis import (
    _first_crossing,
    _interpolate,
    _paired_summary,
    _validate_history,
    atomic_json,
    evaluated_validation_tokens,
    file_sha256,
    load_json,
)
from speck.paper_finalist import materialize_finalist


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_finalist_plan(plan_path, materialization_contract_path):
    plan_path, plan = load_json(plan_path)
    contract_path, contract = load_json(materialization_contract_path)
    repository_root = plan_path.parents[2]
    if (
        plan.get("format") != "speck_paper_finalist_analysis_plan"
        or plan.get("format_version") != 1
        or plan.get("status") != "frozen_after_proxy_pass_before_finalist_materialization"
        or contract.get("format") != "speck_paper_finalist_materialization_contract"
        or contract.get("format_version") != 1
        or contract.get("inputs", {}).get("finalist_analysis")
        != plan_path.relative_to(repository_root).as_posix()
        or contract.get("inputs", {}).get("finalist_analysis_sha256") != file_sha256(plan_path)
        or plan.get("paper_id") != contract.get("paper_id")
        or plan.get("policy_id") != contract.get("policy_id")
    ):
        raise ValueError("finalist analysis and materialization contracts do not match")
    manifest = materialize_finalist(contract_path, check=True)
    materialization_path = repository_root / contract["identity"]["output_root"] / "finalist_materialization.json"
    pairs = [
        {key: value[key] for key in ("pair", "seed", "data_token_offset")}
        for value in contract["pairs"]
    ]
    shared = plan.get("shared_training", {})
    statistical = plan.get("statistical_contract", {})
    stopping = plan.get("stopping_rule", {})
    if (
        plan.get("pairs") != pairs
        or shared.get("training_tokens_per_arm") != contract["materialized_training"]["training_tokens"]
        or shared.get("optimizer_steps") != contract["materialized_training"]["optimizer_steps"]
        or shared.get("evaluation_steps") != [0, 5874, 11748, 17622, 23496]
        or statistical.get("paired_runs") != 6
        or statistical.get("student_t_critical_df_5") != 2.0150483733330233
        or statistical.get("language_loss_non_inferiority_margin_nats") != 0.01
        or statistical.get("source_guardrail_nats") != 0.02
        or stopping.get("required_complete_model_runs") != 12
        or stopping.get("required_control_runs_before_candidates") != 6
        or stopping.get("interim_efficacy_looks") != 0
        or stopping.get("interim_futility_looks") != 0
        or manifest.get("training_authorized") is not False
    ):
        raise ValueError("finalist analysis plan is incomplete")
    return (
        plan_path,
        plan,
        contract_path,
        contract,
        materialization_path,
        manifest,
    )


def collect_run_result(plan_path, materialization_contract_path, experiment, checkpoint_dir=None):
    """Normalize one complete finalist checkpoint into an auditable result."""

    (
        plan_path,
        plan,
        contract_path,
        contract,
        materialization_path,
        _,
    ) = load_finalist_plan(plan_path, materialization_contract_path)
    experiment = Path(experiment).expanduser().resolve()
    configs = load_experiment(experiment, "model", "runtime", "train")
    train = configs["train"]
    arm_id = experiment.name
    arm = contract["parent_arms"].get(arm_id)
    if arm is None:
        raise ValueError("experiment is not a frozen finalist arm")
    pair = next(
        (
            value
            for value in plan["pairs"]
            if value["seed"] == train.get("seed")
            and value["data_token_offset"] == train.get("data_token_offset")
        ),
        None,
    )
    if pair is None:
        raise ValueError("experiment does not match a frozen finalist pair")
    pair_id = f"pair-{pair['pair']}-seed-{pair['seed']}-order-{pair['data_token_offset']}"
    expected_run = f"{contract['identity']['family_id']}-{pair_id}-{arm_id}"
    if train.get("run") != expected_run or experiment.parent.name != pair_id:
        raise ValueError("finalist experiment run identity is invalid")
    shared = plan["shared_training"]
    expected_step = shared["optimizer_steps"]
    checkpoint_dir = (
        Path(checkpoint_dir).expanduser().resolve()
        if checkpoint_dir is not None
        else Path(train["output_dir"]).expanduser().resolve()
    )
    if completed_steps(checkpoint_dir) != [expected_step]:
        raise ValueError("finalist run must retain exactly one complete final checkpoint")
    metadata = load_metadata(checkpoint_dir, expected_step)
    resolved = metadata.get("resolved", {})
    world_size = resolved.get("world_size")
    if world_size != 1 or configs["runtime"].get("device_batch_size") != 4:
        raise ValueError("finalist requires the frozen single-GPU batch geometry")
    validation_batch_tokens = 4 * shared["sequence_length"]
    intermediate_tokens = evaluated_validation_tokens(
        shared["intermediate_validation_tokens"], validation_batch_tokens
    )
    final_tokens = evaluated_validation_tokens(
        shared["final_validation_tokens"], validation_batch_tokens
    )
    expected_resolved = {
        "seed": pair["seed"],
        "data_token_offset": pair["data_token_offset"],
        "train_tokens": shared["training_tokens_per_arm"],
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
        differences = sorted(key for key in expected_resolved if expected_resolved[key] != actual_resolved[key])
        raise ValueError(f"finalist completed run drifted: {differences}")
    if (
        metadata.get("partial")
        or metadata.get("global_tokens") != shared["training_tokens_per_arm"]
        or metadata.get("validation_step") != expected_step
        or metadata.get("validation_tokens") != final_tokens
    ):
        raise ValueError("finalist final checkpoint is incomplete")
    history = metadata.get("validation_history")
    _validate_history(history, shared["evaluation_steps"], intermediate_tokens, final_tokens)
    if any(
        entry["training_tokens"] != entry["step"] * shared["batch_tokens"] for entry in history
    ):
        raise ValueError("finalist validation history token positions changed")
    summary_path = checkpoint_dir / "run_summary.json"
    _, summary = load_json(summary_path)
    if (
        summary.get("partial")
        or summary.get("completed_steps") != expected_step
        or summary.get("global_tokens") != shared["training_tokens_per_arm"]
        or summary.get("validation_history") != history
    ):
        raise ValueError("finalist run summary does not match its final checkpoint")
    timing = load_timing(checkpoint_dir, expected_step)
    if timing is None:
        raise ValueError("finalist run is missing final timing evidence")
    complete_path = checkpoint_dir / f"complete_{expected_step:06d}"
    return {
        "format": "speck_paper_finalist_run_result",
        "format_version": 1,
        "status": "complete_qualified",
        "created_at": _utc_now(),
        "checkpoint_completed_at": datetime.fromtimestamp(
            complete_path.stat().st_mtime, timezone.utc
        ).isoformat(),
        "paper_id": plan["paper_id"],
        "policy_id": plan["policy_id"],
        "analysis_plan_sha256": file_sha256(plan_path),
        "materialization_contract_sha256": file_sha256(contract_path),
        "materialization_sha256": file_sha256(materialization_path),
        "pair": pair,
        "arm_id": arm_id,
        "experiment": str(experiment),
        "run": expected_run,
        "checkpoint": checkpoint_identity(checkpoint_dir, expected_step),
        "run_summary": {"path": str(summary_path), "sha256": file_sha256(summary_path)},
        "parameters": arm["parameters"],
        "flops_per_token_at_4096": arm["flops_per_token_at_4096"],
        "training_tokens": shared["training_tokens_per_arm"],
        "validation_history": history,
        "final_validation": history[-1],
        "timing": timing,
        "peak_allocated_bytes": metadata.get("peak_allocated_bytes"),
        "non_finite_steps": 0,
    }


def _validate_result(report, plan, contract, plan_sha, contract_sha, materialization_sha):
    required = {
        "format",
        "format_version",
        "status",
        "created_at",
        "paper_id",
        "policy_id",
        "analysis_plan_sha256",
        "materialization_contract_sha256",
        "materialization_sha256",
        "pair",
        "arm_id",
        "parameters",
        "flops_per_token_at_4096",
        "training_tokens",
        "validation_history",
        "final_validation",
        "non_finite_steps",
    }
    arm = contract["parent_arms"].get(report.get("arm_id"), {})
    shared = plan["shared_training"]
    if not required <= set(report) or (
        report.get("format") != "speck_paper_finalist_run_result"
        or report.get("format_version") != 1
        or report.get("status") != "complete_qualified"
        or report.get("paper_id") != plan["paper_id"]
        or report.get("policy_id") != plan["policy_id"]
        or report.get("analysis_plan_sha256") != plan_sha
        or report.get("materialization_contract_sha256") != contract_sha
        or report.get("materialization_sha256") != materialization_sha
        or report.get("pair") not in plan["pairs"]
        or report.get("parameters") != arm.get("parameters")
        or report.get("flops_per_token_at_4096") != arm.get("flops_per_token_at_4096")
        or report.get("training_tokens") != shared["training_tokens_per_arm"]
        or report.get("non_finite_steps") != 0
    ):
        raise ValueError("finalist run result identity or geometry is invalid")
    validation_batch_tokens = 4 * shared["sequence_length"]
    _validate_history(
        report["validation_history"],
        shared["evaluation_steps"],
        evaluated_validation_tokens(
            shared["intermediate_validation_tokens"], validation_batch_tokens
        ),
        evaluated_validation_tokens(shared["final_validation_tokens"], validation_batch_tokens),
    )
    if report["final_validation"] != report["validation_history"][-1] or any(
        entry["training_tokens"] != entry["step"] * shared["batch_tokens"]
        for entry in report["validation_history"]
    ):
        raise ValueError("finalist final validation or token cadence changed")


def _load_results(plan_path, materialization_contract_path, result_paths):
    (
        plan_path,
        plan,
        contract_path,
        contract,
        materialization_path,
        _,
    ) = load_finalist_plan(plan_path, materialization_contract_path)
    results = []
    references = []
    for path in result_paths:
        path, report = load_json(path)
        _validate_result(
            report,
            plan,
            contract,
            file_sha256(plan_path),
            file_sha256(contract_path),
            file_sha256(materialization_path),
        )
        results.append(report)
        references.append({"path": str(path), "sha256": file_sha256(path)})
    return plan_path, plan, contract_path, contract, materialization_path, results, references


def lock_time_to_quality_target(plan_path, materialization_contract_path, control_result_paths):
    """Lock the finalist target from exactly six completed controls."""

    plan_path, plan, contract_path, _, materialization_path, results, references = _load_results(
        plan_path, materialization_contract_path, control_result_paths
    )
    control = plan["arms"]["control"]
    if (
        len(results) != 6
        or any(result["arm_id"] != control for result in results)
        or {result["pair"]["pair"] for result in results}
        != {pair["pair"] for pair in plan["pairs"]}
    ):
        raise ValueError("finalist target locking requires exactly all six control results")
    target = math.ceil(
        max(result["final_validation"]["validation_loss"] for result in results) * 1_000_000
    ) / 1_000_000
    return {
        "format": "speck_paper_finalist_time_to_quality_lock",
        "format_version": 1,
        "status": "locked_from_six_controls_before_candidates",
        "locked_at": _utc_now(),
        "paper_id": plan["paper_id"],
        "policy_id": plan["policy_id"],
        "analysis_plan_sha256": file_sha256(plan_path),
        "materialization_contract_sha256": file_sha256(contract_path),
        "materialization_sha256": file_sha256(materialization_path),
        "control_arm": control,
        "control_results": references,
        "rule": plan["control_only_target_lock"]["rule"],
        "round_up_decimals": 6,
        "validation_loss_target": target,
    }


def analyze_finalist(plan_path, materialization_contract_path, target_lock_path, result_paths):
    """Apply the frozen six-pair finalist language analysis."""

    plan_path, plan, contract_path, contract, materialization_path, results, references = _load_results(
        plan_path, materialization_contract_path, result_paths
    )
    target_lock_path, target_lock = load_json(target_lock_path)
    if (
        target_lock.get("format") != "speck_paper_finalist_time_to_quality_lock"
        or target_lock.get("status") != "locked_from_six_controls_before_candidates"
        or target_lock.get("analysis_plan_sha256") != file_sha256(plan_path)
        or target_lock.get("materialization_contract_sha256") != file_sha256(contract_path)
        or target_lock.get("materialization_sha256") != file_sha256(materialization_path)
    ):
        raise ValueError("finalist time-to-quality target lock is invalid")
    arms = plan["arms"]
    expected = {(pair["pair"], arm) for pair in plan["pairs"] for arm in arms.values()}
    observed = {(result["pair"]["pair"], result["arm_id"]) for result in results}
    if len(results) != 12 or observed != expected:
        raise ValueError("finalist analysis requires exactly one result for every frozen cell")
    locked_at = datetime.fromisoformat(target_lock["locked_at"].replace("Z", "+00:00"))
    if any(
        datetime.fromisoformat(result["created_at"].replace("Z", "+00:00")) <= locked_at
        for result in results
        if result["arm_id"] == arms["candidate"]
    ):
        raise ValueError("finalist candidate results must be created after target locking")
    by_cell = {(result["pair"]["pair"], result["arm_id"]): result for result in results}
    t_critical = plan["statistical_contract"]["student_t_critical_df_5"]
    target = target_lock["validation_loss_target"]
    compute = plan["analysis_views"]["fixed_analytic_flops"]
    token_differences = []
    compute_differences = []
    time_differences = []
    time_improvements = []
    censored = []
    paired = []
    for pair in plan["pairs"]:
        control = by_cell[pair["pair"], arms["control"]]
        candidate = by_cell[pair["pair"], arms["candidate"]]
        token_difference = (
            candidate["final_validation"]["validation_loss"]
            - control["final_validation"]["validation_loss"]
        )
        compute_difference = _interpolate(
            candidate["validation_history"],
            "training_tokens",
            compute["candidate_training_tokens"],
        ) - _interpolate(
            control["validation_history"],
            "training_tokens",
            compute["control_training_tokens"],
        )
        time_budget = min(
            control["final_validation"]["steady_training_seconds"],
            candidate["final_validation"]["steady_training_seconds"],
        )
        time_difference = _interpolate(
            candidate["validation_history"], "steady_training_seconds", time_budget
        ) - _interpolate(control["validation_history"], "steady_training_seconds", time_budget)
        control_time = _first_crossing(control["validation_history"], target)
        candidate_time = _first_crossing(candidate["validation_history"], target)
        time_result = {
            "control_seconds": control_time,
            "candidate_seconds": candidate_time,
            "right_censored": control_time is None or candidate_time is None,
        }
        if time_result["right_censored"]:
            censored.append(pair["pair"])
        else:
            improvement = 1 - candidate_time / control_time if control_time else 0.0
            time_result["candidate_relative_improvement"] = improvement
            time_improvements.append(improvement)
        paired.append(
            {
                "pair": pair,
                "fixed_tokens_candidate_minus_control_loss": token_difference,
                "fixed_flops_candidate_minus_control_loss": compute_difference,
                "fixed_steady_time_seconds": time_budget,
                "fixed_steady_time_candidate_minus_control_loss": time_difference,
                "time_to_quality": time_result,
            }
        )
        token_differences.append(token_difference)
        compute_differences.append(compute_difference)
        time_differences.append(time_difference)
    source_names = set.intersection(
        *(set(result["final_validation"]["validation_source_losses"]) for result in results)
    )
    if not source_names or any(
        set(result["final_validation"]["validation_source_losses"]) != source_names
        for result in results
    ):
        raise ValueError("finalist source guardrails require identical non-empty sources")
    source_guardrails = {}
    source_margin = plan["statistical_contract"]["source_guardrail_nats"]
    for source in sorted(source_names):
        differences = [
            by_cell[pair["pair"], arms["candidate"]]["final_validation"][
                "validation_source_losses"
            ][source]
            - by_cell[pair["pair"], arms["control"]]["final_validation"][
                "validation_source_losses"
            ][source]
            for pair in plan["pairs"]
        ]
        summary = _paired_summary(differences, t_critical)
        summary["margin_nats"] = source_margin
        summary["pass"] = summary["upper_one_sided_95_bound"] <= source_margin
        source_guardrails[source] = summary
    fixed_tokens = _paired_summary(token_differences, t_critical)
    fixed_tokens["margin_nats"] = plan["statistical_contract"][
        "language_loss_non_inferiority_margin_nats"
    ]
    fixed_tokens["non_inferiority_pass"] = (
        fixed_tokens["upper_one_sided_95_bound"] <= fixed_tokens["margin_nats"]
    )
    all_sources = all(value["pass"] for value in source_guardrails.values())
    return {
        "format": "speck_paper_finalist_analysis",
        "format_version": 1,
        "status": "complete_finalist_language_evidence_no_standalone_promotion",
        "created_at": _utc_now(),
        "paper_id": plan["paper_id"],
        "policy_id": plan["policy_id"],
        "analysis_plan_sha256": file_sha256(plan_path),
        "materialization_contract_sha256": file_sha256(contract_path),
        "materialization_sha256": file_sha256(materialization_path),
        "time_to_quality_lock": {
            "path": str(target_lock_path),
            "sha256": file_sha256(target_lock_path),
            "validation_loss_target": target,
        },
        "run_results": references,
        "paired_results": paired,
        "fixed_tokens": fixed_tokens,
        "fixed_analytic_flops": _paired_summary(compute_differences, t_critical),
        "fixed_steady_training_time": _paired_summary(time_differences, t_critical),
        "time_to_quality": {
            "right_censored_pairs": censored,
            "paired_relative_improvement": (
                _paired_summary(time_improvements, t_critical) if not censored else None
            ),
        },
        "source_guardrails": source_guardrails,
        "finalist_language_screen_pass": fixed_tokens["non_inferiority_pass"] and all_sources,
        "authority": plan["decision_authority"],
    }


__all__ = [
    "analyze_finalist",
    "atomic_json",
    "collect_run_result",
    "file_sha256",
    "load_finalist_plan",
    "lock_time_to_quality_target",
]
