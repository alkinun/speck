"""Validate the append-only finalist systems-measurement claim boundary."""

import argparse
import hashlib
import json
from pathlib import Path


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    return parser.parse_args(argv)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_audit(audit, root):
    _require(
        audit.get("format") == "speck_paper_finalist_systems_measurement_audit"
        and audit.get("format_version") == 1
        and audit.get("status")
        == "timing_and_peak_allocation_descriptive_energy_and_causal_systems_claims_blocked",
        "invalid finalist systems-measurement audit identity",
    )
    scope = audit.get("scope", "")
    _require(
        scope.startswith("blind static audit")
        and all(
            boundary in scope
            for boundary in (
                "no live GPU was queried",
                "no service was polled",
                "no checkpoint, log, result, or intermediate metric was accessed",
                "no runtime environment was modified",
            )
        ),
        "finalist systems-measurement scope changed",
    )
    loaded = {}
    for key, reference in audit.get("frozen_inputs", {}).items():
        path = root / reference.get("path", "")
        _require(
            path.is_file() and file_sha256(path) == reference.get("sha256"),
            f"finalist systems-measurement {key} input changed",
        )
        if path.suffix == ".json":
            loaded[key] = json.loads(path.read_text(encoding="utf-8"))

    order = loaded["automation"]["execution_order"]
    execution = audit.get("execution_design", {})
    _require(
        len(order) == 12
        and all(name.endswith("dense_global_param_match") for name in order[:6])
        and all(name.endswith("five_cache_kda_gqa") for name in order[6:])
        and execution.get("runs") == 12
        and execution.get("forecast_steady_gpu_hours")
        == loaded["launch_contract"]["unchanged_execution"]["total_steady_gpu_hours_forecast"]
        and execution.get("control_first_required_for_target_integrity") is True
        and execution.get("arms_interleaved") is False
        and execution.get("arm_order_randomized") is False
        and execution.get("calendar_time_blocked_by_arm") is True
        and execution.get("same_gpu_uuid_required_before_every_launch") is True
        and execution.get("idle_gpu_required_before_every_launch") is True
        and execution.get("maximum_start_temperature_c") == 50
        and execution.get("successor_live_gate_values_persisted") is False,
        "finalist systems execution design changed",
    )

    preserved = audit.get("preserved_measurements", {})
    _require(
        preserved.get("analytic_flops_per_token") is True
        and preserved.get("optimizer_seconds", {}).get("preserved") is True
        and preserved.get("optimizer_seconds", {}).get("cuda_synchronized_at_flush_boundaries")
        is True
        and preserved.get("steady_training_seconds", {}).get("preserved") is True
        and preserved.get("steady_training_seconds", {}).get("excludes_first_10_startup_steps")
        is True
        and preserved.get("steady_training_seconds", {}).get(
            "excludes_validation_and_checkpoint_intervals"
        )
        is True
        and preserved.get("peak_allocated_bytes", {}).get("preserved") is True
        and preserved.get("peak_allocated_bytes", {}).get("peak_reserved_bytes_preserved_per_run")
        is False
        and preserved.get("validation_history_timing_points") == 5
        and preserved.get("final_checkpoint_hashes") is True
        and preserved.get("final_timing_values") is True
        and preserved.get("final_timing_source_file_hash") is False,
        "finalist preserved systems measurements changed",
    )
    missing = audit.get("missing_measurements", {})
    _require(
        len(missing) == 18 and all(value is False for value in missing.values()),
        "finalist missing systems measurements changed",
    )

    confounding = audit.get("temporal_confounding", {})
    _require(
        confounding.get("architecture_and_calendar_block_are_aliased") is True
        and confounding.get("paired_seed_and_data_order_remove_calendar_confounding") is False
        and confounding.get("same_gpu_and_start_temperature_gate_reduce_confounding") is True
        and confounding.get("same_gpu_and_start_temperature_gate_eliminate_confounding") is False
        and confounding.get("quality_endpoint_affected") is False
        and confounding.get("analytic_flops_affected") is False
        and confounding.get("steady_time_and_time_to_quality_causal_attribution_affected") is True
        and confounding.get("energy_and_dollar_attribution_affected") is True,
        "finalist temporal-confounding disposition changed",
    )
    claims = audit.get("claim_boundary", {})
    _require(
        len(claims.get("may_report", ())) == 6
        and len(claims.get("may_not_report", ())) == 7
        and claims.get("finalist_language_quality_authority_unchanged") is True
        and claims.get("secondary_timing_values_must_be_labeled_descriptive") is True
        and claims.get("peak_allocation_is_not_persistent_serving_state") is True,
        "finalist systems claim boundary changed",
    )
    future = audit.get("future_systems_protocol", {})
    _require(
        len(future.get("minimum_design", ())) == 9
        and future.get("may_reuse_language_results_for_systems_randomization") is False
        and future.get("systems_repeats_are_new_prospective_measurements") is True,
        "future finalist systems protocol changed",
    )
    decision = audit.get("decision", {})
    _require(
        decision.get("finalist_language_sequence_remains_authorized") is True
        and decision.get("analytic_flops_reporting_authorized") is True
        and decision.get("on_host_timing_reporting_authorized_as_descriptive") is True
        and decision.get("peak_allocation_reporting_authorized_as_descriptive") is True
        and all(
            decision.get(field) is False
            for field in (
                "causal_training_speedup_claim_authorized",
                "energy_claim_authorized",
                "monetary_claim_authorized",
                "serving_claim_authorized",
                "systems_architecture_promotion_authorized",
                "telemetry_injection_into_active_runs_authorized",
                "execution_order_change_authorized",
                "trainer_runner_or_analysis_modified",
            )
        ),
        "finalist systems fail-closed decision changed",
    )
    return {
        "status": "valid",
        "runs": 12,
        "forecast_steady_gpu_hours": execution["forecast_steady_gpu_hours"],
        "energy_measured": False,
        "causal_systems_claim_authorized": False,
        "language_sequence_authorized": True,
    }


def validate_file(path):
    path = Path(path).resolve()
    return validate_audit(json.loads(path.read_text(encoding="utf-8")), path.parents[2])


def main(argv=None):
    report = validate_file(arguments(argv).audit)
    print(
        "Finalist systems audit: "
        f"{report['status']} ({report['runs']} runs, "
        f"energy={str(report['energy_measured']).lower()})"
    )


if __name__ == "__main__":
    main()
