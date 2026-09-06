"""Analyze the frozen six-block Paper 1 finalist systems protocol."""

import hashlib
import itertools
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path):
    path = Path(path).expanduser().resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return path, value


def atomic_json(path, value):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"systems trial has invalid {name}")
    if positive and value <= 0:
        raise ValueError(f"systems trial requires positive {name}")
    return float(value)


def _energy_interval(trial, name, *, positive_estimate=True, nonnegative_bounds=False):
    value = trial.get(name, {})
    lower = _number(value.get("lower"), f"{name} lower")
    estimate = _number(value.get("estimate"), f"{name} estimate")
    upper = _number(value.get("upper"), f"{name} upper")
    if (
        lower > estimate
        or estimate > upper
        or (positive_estimate and estimate <= 0)
        or (nonnegative_bounds and (lower < 0 or upper <= 0))
    ):
        raise ValueError(f"systems trial has invalid {name} bounds")
    return {"lower": lower, "estimate": estimate, "upper": upper}


def _validate_trial(trial, arm_id, workload, thermal):
    if (
        trial.get("status") != "complete_qualified"
        or trial.get("arm_id") != arm_id
        or trial.get("measured_optimizer_steps") != workload["measured_optimizer_steps"]
        or trial.get("measured_tokens") != workload["measured_tokens_per_trial"]
        or trial.get("non_finite_steps") != 0
        or trial.get("OOM") is not False
        or trial.get("kernel_fallback") is not False
    ):
        raise ValueError("systems trial identity or failure state is invalid")
    wall = _number(trial.get("measured_wall_seconds"), "measured wall seconds", positive=True)
    gross = _energy_interval(
        trial,
        "gross_board_energy_joules",
        nonnegative_bounds=True,
    )
    incremental = _energy_interval(
        trial,
        "incremental_board_energy_joules",
        positive_estimate=False,
    )
    telemetry = trial.get("telemetry", {})
    coverage = _number(telemetry.get("measured_window_coverage"), "telemetry coverage")
    gap = _number(telemetry.get("maximum_sample_gap_seconds"), "telemetry sample gap")
    start_temperature = _number(trial.get("start_temperature_c"), "start temperature")
    if (
        coverage < 0.99
        or coverage > 1
        or gap > 2.5
        or start_temperature > thermal["maximum_trial_start_temperature_c"]
    ):
        raise ValueError("systems trial telemetry or thermal gate is invalid")
    return {
        "wall": wall,
        "gross": gross,
        "incremental": incremental,
        "start_temperature": start_temperature,
        "peak_allocated_bytes": _number(
            trial.get("peak_allocated_bytes"), "peak allocated bytes", positive=True
        ),
        "peak_reserved_bytes": _number(
            trial.get("peak_reserved_bytes"), "peak reserved bytes", positive=True
        ),
        "peak_nvml_used_bytes": _number(
            trial.get("peak_nvml_used_bytes"), "peak NVML used bytes", positive=True
        ),
    }


def _validate_block(report, expected, protocol_sha, protocol):
    if (
        report.get("format") != "speck_paper_finalist_systems_block_result"
        or report.get("format_version") != 1
        or report.get("protocol_sha256") != protocol_sha
        or report.get("block") != expected["block"]
        or report.get("pair") != expected["pair"]
        or report.get("trial_order") != expected["trial_order"]
    ):
        raise ValueError("systems block result does not match the frozen protocol")
    if report.get("status") == "failed_retained":
        if not report.get("failure"):
            raise ValueError("failed systems block has no retained failure")
        return None
    if report.get("status") != "complete_qualified":
        raise ValueError("systems block result has an invalid status")
    trials = report.get("trials", {})
    if set(trials) != {"control", "candidate"}:
        raise ValueError("systems block must contain both trials")
    arms = protocol["arms"]
    workload = protocol["trial_workload"]
    thermal = protocol["thermal_and_idle_design"]
    control = _validate_trial(trials["control"], arms["control"], workload, thermal)
    candidate = _validate_trial(trials["candidate"], arms["candidate"], workload, thermal)
    if (
        abs(control["start_temperature"] - candidate["start_temperature"])
        > thermal["maximum_within_pair_start_temperature_difference_c"]
    ):
        raise ValueError("systems block violates paired start-temperature matching")
    return {"control": control, "candidate": candidate}


def _sign_flip_p(effects):
    observed = statistics.mean(effects)
    permutations = (
        statistics.mean(sign * effect for sign, effect in zip(signs, effects))
        for signs in itertools.product((-1, 1), repeat=len(effects))
    )
    return sum(value <= observed + 1e-15 for value in permutations) / (2 ** len(effects))


def _log_ratio(numerator, denominator):
    if numerator <= 0 or denominator <= 0:
        return None
    return math.log(numerator / denominator)


def _summary(effects_by_block, order_by_block, critical, expected_blocks=6):
    blocks = sorted(effects_by_block)
    effects = [effects_by_block[block] for block in blocks]
    order_effects = {
        order: [effects_by_block[block] for block in blocks if order_by_block[block] == order]
        for order in ("control_then_candidate", "candidate_then_control")
    }
    means = {
        order: statistics.mean(values) if values else None
        for order, values in order_effects.items()
    }
    mean = statistics.mean(effects) if effects else None
    standard_deviation = statistics.stdev(effects) if len(effects) > 1 else None
    upper = (
        mean + critical * standard_deviation / math.sqrt(len(effects))
        if standard_deviation is not None
        else None
    )
    exact_p = _sign_flip_p(effects) if len(effects) == expected_blocks else None
    passed = (
        len(effects) == expected_blocks
        and upper is not None
        and upper < 0
        and exact_p <= 0.05
        and all(value is not None and value < 0 for value in means.values())
    )
    return {
        "n": len(effects),
        "blocks": blocks,
        "log_candidate_over_control": effects,
        "mean_log_ratio": mean,
        "geometric_mean_ratio": math.exp(mean) if mean is not None else None,
        "standard_deviation_log_ratio": standard_deviation,
        "upper_one_sided_97_5_log_ratio": upper,
        "upper_one_sided_97_5_ratio": math.exp(upper) if upper is not None else None,
        "exact_sign_flip_p": exact_p,
        "order_mean_log_ratio": means,
        "pass": passed,
    }


def analyze_systems(protocol_path, result_paths):
    protocol_path, protocol = load_object(protocol_path)
    if (
        protocol.get("format") != "speck_paper_finalist_systems_protocol"
        or protocol.get("format_version") != 2
        or protocol.get("status") != "frozen_before_any_finalist_result_execution_blocked"
    ):
        raise ValueError("systems analysis requires the frozen v2 protocol")
    protocol_sha = file_sha256(protocol_path)
    expected = {block["block"]: block for block in protocol["paired_blocks"]}
    observed = {}
    references = []
    for result_path in result_paths:
        path, report = load_object(result_path)
        block = report.get("block")
        if block not in expected or block in observed:
            raise ValueError("systems results contain an unexpected or duplicate block")
        observed[block] = _validate_block(report, expected[block], protocol_sha, protocol)
        references.append({"path": str(path), "sha256": file_sha256(path), "block": block})
    missing = sorted(set(expected) - set(observed))
    failed = sorted(block for block, value in observed.items() if value is None)
    valid = {block: value for block, value in observed.items() if block not in failed}
    order_by_block = {
        block: (
            "control_then_candidate"
            if expected[block]["trial_order"][0] == "control"
            else "candidate_then_control"
        )
        for block in valid
    }
    time_effects = {
        block: math.log(value["candidate"]["wall"] / value["control"]["wall"])
        for block, value in valid.items()
    }
    gross_conservative_effects = {
        block: effect
        for block, value in valid.items()
        if (
            effect := _log_ratio(
                value["candidate"]["gross"]["upper"],
                value["control"]["gross"]["lower"],
            )
        )
        is not None
    }
    gross_point_effects = {
        block: math.log(
            value["candidate"]["gross"]["estimate"] / value["control"]["gross"]["estimate"]
        )
        for block, value in valid.items()
    }
    incremental_effects = {
        block: effect
        for block, value in valid.items()
        if (
            effect := _log_ratio(
                value["candidate"]["incremental"]["estimate"],
                value["control"]["incremental"]["estimate"],
            )
        )
        is not None
    }
    critical = protocol["estimands_and_analysis"]["student_t_critical"]
    primary_time = _summary(time_effects, order_by_block, critical)
    primary_energy = _summary(gross_conservative_effects, order_by_block, critical)
    gross_point = _summary(gross_point_effects, order_by_block, critical)
    incremental = _summary(incremental_effects, order_by_block, critical)
    complete = not missing and not failed and len(valid) == 6
    joint = (
        complete
        and primary_time["pass"]
        and primary_energy["pass"]
        and incremental["n"] == 6
        and incremental["mean_log_ratio"] < 0
    )
    return {
        "format": "speck_paper_finalist_systems_analysis",
        "format_version": 1,
        "status": (
            "incomplete_failed_no_training_systems_claim"
            if not complete
            else (
                "complete_joint_training_systems_efficiency_pass"
                if joint
                else "complete_no_joint_training_systems_efficiency_claim"
            )
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {"path": str(protocol_path), "sha256": protocol_sha},
        "result_references": sorted(references, key=lambda value: value["block"]),
        "missing_blocks": missing,
        "failed_blocks": failed,
        "valid_blocks": sorted(valid),
        "primary": {
            "measured_wall_seconds_per_token": primary_time,
            "gross_board_joules_per_token_conservative": primary_energy,
            "gross_board_joules_per_token_point": gross_point,
        },
        "sensitivity": {"incremental_board_joules_per_token_point": incremental},
        "joint_training_systems_efficiency_pass": joint,
        "failure_policy": {
            "imputation": False,
            "outlier_deletion": False,
            "replacement_block": False,
            "automatic_retry": False,
        },
        "claim_boundary": {
            "same_RTX_3090_4K_BF16_training_step_time": complete and primary_time["pass"],
            "gross_GPU_board_energy": complete and primary_energy["pass"],
            "whole_system_energy": False,
            "monetary": False,
            "serving": False,
            "language_quality": False,
            "component_attribution": False,
            "paper_scale": False,
        },
    }
