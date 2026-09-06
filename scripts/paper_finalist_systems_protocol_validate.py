"""Validate the pre-result thermally interleaved finalist systems protocol."""

import argparse
import hashlib
import json
from pathlib import Path


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    return parser.parse_args(argv)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_protocol(protocol, root):
    require(
        protocol.get("format") == "speck_paper_finalist_systems_protocol"
        and protocol.get("format_version") == 2
        and protocol.get("protocol_id") == "speck-paper-1-finalist-systems-v2"
        and protocol.get("status") == "frozen_before_any_finalist_result_execution_blocked",
        "invalid finalist systems protocol identity",
    )
    boundary = protocol.get("pre_result_boundary", {})
    require(
        boundary.get("accepted_finalist_results") == 0
        and all(
            boundary.get(field) is False
            for field in (
                "active_result_or_checkpoint_inspected",
                "active_service_polled",
                "language_outcomes_used_to_choose_design",
                "telemetry_injected_into_active_sequence",
                "active_runner_trainer_analyzer_plan_or_program_changed",
            )
        ),
        "finalist systems pre-result boundary changed",
    )
    loaded = {}
    for name, reference in protocol.get("inputs", {}).items():
        path = root / reference.get("path", "")
        require(
            path.is_file() and file_sha256(path) == reference.get("sha256"),
            f"finalist systems input changed: {name}",
        )
        loaded[name] = json.loads(path.read_text(encoding="utf-8"))
    require(
        len(loaded) == 7
        and loaded["measurement_boundary"]["decision"][
            "telemetry_injection_into_active_runs_authorized"
        ]
        is False
        and loaded["systems_cost_readiness"]["statistical_controls"]["microbenchmark"].startswith(
            "at least five thermally interleaved"
        ),
        "finalist systems predecessor requirements changed",
    )

    activation = protocol.get("activation_gate", {})
    require(
        activation.get("accepted_results_required") == 12
        and activation.get("v3_result_acceptance_required") is True
        and activation.get("all_12_final_checkpoints_retained") is True
        and activation.get("systems_harness_qualified") is True
        and activation.get("systems_analysis_qualified") is True
        and activation.get("current_activation_gate_pass") is False,
        "finalist systems activation gate changed",
    )
    hardware = protocol.get("hardware_and_software_lock", {})
    require(
        hardware.get("gpu_uuid") == "GPU-6e7f2d05-ac19-3812-ff41-33079fd67bfd"
        and hardware.get("gpu_count") == 1
        and hardware.get("precision") == "bfloat16"
        and hardware.get("device_batch_size") == 4
        and hardware.get("sequence_length") == 4096
        and hardware.get("batch_tokens") == 65536
        and hardware.get("power_limit_change_authorized") is False
        and hardware.get("clock_lock_change_authorized") is False
        and hardware.get("software_or_kernel_change_within_protocol_authorized") is False,
        "finalist systems hardware lock changed",
    )

    expected_pairs = loaded["materialization_contract"]["pairs"]
    blocks = protocol.get("paired_blocks", [])
    expected_orders = [
        ["control", "candidate"],
        ["candidate", "control"],
        ["candidate", "control"],
        ["control", "candidate"],
        ["control", "candidate"],
        ["candidate", "control"],
    ]
    require(
        len(blocks) == 6
        and [block.get("block") for block in blocks] == list(range(6))
        and [block.get("pair") for block in blocks] == list(range(6))
        and [block.get("trial_order") for block in blocks] == expected_orders,
        "finalist systems paired-block order changed",
    )
    for block, pair in zip(blocks, expected_pairs):
        require(
            block.get("seed") == pair["seed"]
            and block.get("data_token_offset") == pair["data_token_offset"]
            and block.get("benchmark_data_start") == pair["end_token_offset"]
            and block.get("benchmark_data_end") - block.get("benchmark_data_start") == 2_621_440,
            "finalist systems pair or benchmark data window changed",
        )
    balance = protocol.get("balance_rationale", {})
    require(
        balance.get("blocks") == 6
        and balance.get("control_first_blocks") == 3
        and balance.get("candidate_first_blocks") == 3
        and balance.get("each_seed_has_one_control_first_and_one_candidate_first") is True
        and balance.get("maximum_consecutive_same_order") == 2
        and balance.get("order_frozen_without_language_results") is True,
        "finalist systems balance changed",
    )

    workload = protocol.get("trial_workload", {})
    require(
        workload.get("checkpoint_step") == 23496
        and workload.get("checkpoint_mutation_or_rewrite") is False
        and workload.get("warmup_optimizer_steps") == 10
        and workload.get("measured_optimizer_steps") == 30
        and workload.get("total_optimizer_steps_per_trial") == 40
        and workload.get("total_tokens_per_trial") == 2_621_440
        and workload.get("measured_tokens_per_trial") == 1_966_080
        and workload.get("all_trial_total_tokens") == 31_457_280
        and workload.get("all_trial_measured_tokens") == 23_592_960
        and workload.get("validation_during_trial") is False
        and workload.get("checkpoint_write_during_trial") is False
        and workload.get("model_or_optimizer_output_persisted") is False
        and workload.get("loss_used_for_decision") is False,
        "finalist systems trial workload changed",
    )
    thermal = protocol.get("thermal_and_idle_design", {})
    require(
        thermal.get("pre_block_idle_seconds") == 120
        and thermal.get("post_block_idle_seconds") == 120
        and thermal.get("idle_blocks_before") == 6
        and thermal.get("idle_blocks_after") == 6
        and thermal.get("maximum_trial_start_temperature_c") == 45
        and thermal.get("maximum_within_pair_start_temperature_difference_c") == 2
        and thermal.get("thermal_recovery_timeout_seconds") == 1200
        and thermal.get("other_compute_processes_at_start") == 0,
        "finalist systems thermal design changed",
    )

    telemetry = protocol.get("telemetry", {})
    gpu_fields = {
        "power.draw",
        "temperature.gpu",
        "clocks.current.graphics",
        "clocks.current.memory",
        "power.limit",
        "pstate",
        "utilization.gpu",
        "utilization.memory",
        "fan.speed",
        "clocks_throttle_reasons.active",
    }
    require(
        telemetry.get("sampling_hz") == 1
        and set(telemetry.get("gpu_fields", ())) == gpu_fields
        and len(telemetry.get("host_fields", ())) == 7
        and len(telemetry.get("trial_phase_markers", ())) == 8
        and telemetry.get("minimum_measured_window_coverage") == 0.99
        and telemetry.get("maximum_sample_gap_seconds") == 2.5
        and telemetry.get("gap_lower_bound") == "integrate missing duration at zero board watts"
        and telemetry.get("gap_upper_bound")
        == "integrate missing duration at the maximum recorded board power limit spanning the gap"
        and "candidate upper energy divided by control lower energy"
        in telemetry.get("energy_claim_gap_rule", "")
        and telemetry.get("unbounded_outage_interpolation_authorized") is False
        and telemetry.get("whole_system_energy_claim_authorized") is False,
        "finalist systems telemetry contract changed",
    )
    timing = protocol.get("timing_and_memory", {})
    require(
        timing.get("primary_time", "").startswith("CUDA-synchronized monotonic wall seconds")
        and len(timing.get("secondary_time", ())) == 7
        and len(timing.get("memory", ())) == 5
        and timing.get("cuda_synchronize_before_and_after_every_interval") is True
        and timing.get("compile_or_kernel_fallback_retained_as_failure") is True
        and timing.get("OOM_retained_as_failure") is True,
        "finalist systems timing or memory contract changed",
    )

    analysis = protocol.get("estimands_and_analysis", {})
    require(
        len(analysis.get("primary_endpoints", ())) == 2
        and len(analysis.get("sensitivity_endpoints", ())) == 4
        and analysis.get("block_effect") == "natural logarithm of candidate divided by control"
        and analysis.get("one_sided_confidence") == 0.975
        and analysis.get("student_t_df") == 5
        and analysis.get("student_t_critical") == 2.570581835636314
        and analysis.get("exact_sign_flip_assignments") == 64
        and analysis.get("maximum_exact_one_sided_p") == 0.05
        and "AB and BA point estimates are both below zero"
        in analysis.get("metric_specific_improvement_rule", "")
        and "candidate-upper/control-lower" in analysis.get("metric_specific_improvement_rule", "")
        and "both primary endpoints pass"
        in analysis.get("joint_training_systems_efficiency_rule", "")
        and analysis.get("individual_trials_and_all_telemetry_published") is True
        and analysis.get("quality_or_language_metric_analyzed") is False,
        "finalist systems estimand or analysis changed",
    )
    resource = protocol.get("resource_envelope", {})
    require(
        resource.get("trials") == 12
        and resource.get("optimizer_steps") == 480
        and resource.get("measured_optimizer_steps") == 360
        and resource.get("maximum_total_elapsed_hours") == 4
        and resource.get("automatic_budget_extension") is False
        and resource.get("checkpoint_storage_growth_bytes") == 0,
        "finalist systems resource envelope changed",
    )
    implementation = protocol.get("implementation_gate", {})
    require(
        all(
            implementation.get(field) is False
            for field in (
                "systems_harness_present",
                "systems_harness_tests_pass",
                "telemetry_sampler_qualified",
                "energy_integrator_qualified",
                "analysis_implementation_qualified",
                "execution_authorized",
            )
        ),
        "finalist systems implementation gate changed",
    )
    claims = protocol.get("claim_boundary", {})
    require(
        len(claims.get("may_support_after_passing", ())) == 4
        and len(claims.get("may_not_support", ())) == 6,
        "finalist systems claim boundary changed",
    )
    return {
        "status": "valid_frozen_execution_blocked",
        "blocks": len(blocks),
        "trials": resource["trials"],
        "measured_steps": resource["measured_optimizer_steps"],
        "balanced_orders": True,
        "activation_gate_pass": False,
    }


def validate_file(path):
    path = Path(path).resolve()
    return validate_protocol(json.loads(path.read_text(encoding="utf-8")), path.parents[2])


def main(argv=None):
    report = validate_file(arguments(argv).protocol)
    print(
        "Finalist systems protocol: "
        f"{report['status']} ({report['blocks']} blocks, {report['trials']} trials, "
        f"activation={str(report['activation_gate_pass']).lower()})"
    )


if __name__ == "__main__":
    main()
