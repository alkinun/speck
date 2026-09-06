"""Validate frozen QB finite-batch tie policies on discovery and unseen cells."""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from speck.quantile_balancing_tie_reference import POLICIES, fixed_score_replay

ROOT = Path(__file__).resolve().parents[1]


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_json(path, value):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def repository_revision():
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout
    if status.strip():
        raise ValueError("QB tie validation requires a clean repository")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def validate_v1(protocol, root=ROOT):
    if (
        protocol.get("format") != "speck_quantile_balancing_tie_readiness"
        or protocol.get("format_version") != 1
        or protocol.get("status")
        != "discovery_observed_unseen_validation_frozen_training_blocked"
    ):
        raise ValueError("invalid QB tie readiness identity")
    for reference in protocol.get("evidence", {}).values():
        path = root / reference.get("path", "")
        if not path.is_file() or file_sha256(path) != reference.get("sha256"):
            raise ValueError("QB tie readiness evidence changed")
    candidates = protocol.get("candidate_policies", [])
    future = protocol.get("required_future_training_gate", {})
    boundary = protocol.get("boundary", {})
    if (
        [candidate.get("id") for candidate in candidates] != list(POLICIES)
        or [candidate.get("selection_eligible") for candidate in candidates]
        != [False, False, True, False]
        or future.get("source_upper_histogram_and_midpoint_arms_required") is not True
        or future.get("decision") != "a fixed-score CPU pass cannot choose a training policy"
        or boundary.get("active_architecture_or_model_code_changed") is not False
        or boundary.get("active_experiment_program_changed") is not False
        or boundary.get("active_result_checkpoint_service_GPU_or_runtime_accessed") is not False
        or boundary.get("model_integration_authorized") is not False
        or boundary.get("training_authorized") is not False
        or boundary.get("promotion_authority") is not False
    ):
        raise ValueError("QB tie validation contract changed")
    return protocol


def validate_protocol(protocol, root=ROOT):
    if (
        protocol.get("format") != "speck_quantile_balancing_tie_readiness"
        or protocol.get("format_version") != 2
        or protocol.get("status")
        != "corrected_subgradient_bracket_fresh_unseen_validation_frozen_training_blocked"
    ):
        raise ValueError("invalid QB tie readiness v2 identity")
    predecessor = protocol.get("predecessor", {})
    predecessor_path = root / predecessor.get("path", "")
    if (
        not predecessor_path.is_file()
        or file_sha256(predecessor_path) != predecessor.get("sha256")
        or predecessor.get("preserved_append_only") is not True
        or predecessor.get("qualification_authority") is not False
    ):
        raise ValueError("QB tie v1 predecessor changed")
    base = validate_v1(load_object(predecessor_path), root)
    correction = protocol.get("correction", {})
    consumed = protocol.get("consumed_v1_holdout", {})
    preserved = protocol.get("preserved_from_v1", {})
    validation = protocol.get("fresh_unseen_validation", {})
    gate = validation.get("candidate_gate", {})
    future = protocol.get("required_future_training_gate", {})
    boundary = protocol.get("boundary", {})
    if (
        correction.get("v1_invalid_condition") != "count(r < tau) <= q < count(r <= tau)"
        or correction.get("v2_coordinate_subgradient_condition")
        != "count(r < tau) <= q <= count(r <= tau)"
        or correction.get("v1_result_artifact_created") is not False
        or consumed.get("seeds_per_shape") != "32 through 95 inclusive"
        or consumed.get("cases") != 256
        or consumed.get("selection_or_qualification_authority") is not False
        or not all(preserved.get(key) is True for key in (
            "evidence",
            "four_shapes",
            "score_and_initial_bias_distribution",
            "candidate_policies_and_roles",
            "fixed_score_scope",
            "future_training_blockers",
        ))
        or preserved.get("maximum_updates") != 50
        or validation.get("frozen_before_output") is not True
        or validation.get("seeds_per_shape") != "96 through 159 inclusive"
        or validation.get("cases") != 256
        or validation.get("maximum_updates") != 50
        or gate.get("policy") != "interval_midpoint"
        or gate.get("perfect_balance_cases_required") != 256
        or gate.get("cycles_allowed") != 0
        or gate.get("unresolved_at_50_allowed") != 0
        or gate.get("coordinate_subgradient_failures_allowed") != 0
        or gate.get("coordinate_interval_violations_allowed") != 0
        or gate.get("nonfinite_values_allowed") != 0
        or future != base.get("required_future_training_gate")
        or boundary.get("active_architecture_or_model_code_changed") is not False
        or boundary.get("active_experiment_program_changed") is not False
        or boundary.get("active_result_checkpoint_service_GPU_or_runtime_accessed") is not False
        or boundary.get("v1_holdout_reused_for_v2_qualification") is not False
        or boundary.get("model_integration_authorized") is not False
        or boundary.get("training_authorized") is not False
        or boundary.get("promotion_authority") is not False
    ):
        raise ValueError("QB tie v2 validation contract changed")
    return {"base": base, "validation": validation, "consumed": consumed}


def score_case(shape_index, shape, seed):
    generator = torch.Generator().manual_seed(200_000 * shape_index + seed)
    scores = torch.rand(
        (shape["tokens"], shape["experts"]), generator=generator, dtype=torch.float64
    )
    bias = torch.randn(shape["experts"], generator=generator, dtype=torch.float64) * 0.15
    return scores, bias - bias.mean()


def summarize_policy(protocol, policy, seeds, maximum_updates):
    records = []
    for shape_index, shape in enumerate(protocol["observed_discovery"]["shapes"]):
        target = shape["tokens"] * shape["selected"] // shape["experts"]
        for seed in seeds:
            scores, bias = score_case(shape_index, shape, seed)
            result = fixed_score_replay(
                scores, bias, shape["selected"], policy, maximum_updates
            )
            initial_error = (result["initial_loads"] - target).abs()
            final_error = (result["final_loads"] - target).abs()
            records.append(
                {
                    "status": result["status"],
                    "updates": result["updates"],
                    "initial_maximum_load_error": int(initial_error.max().item()),
                    "final_maximum_load_error": int(final_error.max().item()),
                    "initial_mean_load_error": initial_error.to(torch.float64).mean().item(),
                    "final_mean_load_error": final_error.to(torch.float64).mean().item(),
                    "coordinate_subgradient_failures": result[
                        "coordinate_subgradient_failures"
                    ],
                    "coordinate_interval_violations": result[
                        "coordinate_interval_violations"
                    ],
                    "nonfinite_values": result["nonfinite_values"],
                    "maximum_absolute_bias": result["maximum_absolute_bias"],
                    "route_churn": result["route_churn"],
                }
            )
    successful_updates = [
        record["updates"] for record in records if record["status"] == "perfect_balance"
    ]
    churn = [value for record in records for value in record["route_churn"]]
    return {
        "cases": len(records),
        "perfect_balance_cases": len(successful_updates),
        "cycle_cases": sum(record["status"] == "cycle" for record in records),
        "unresolved_cases": sum(record["status"] == "unresolved" for record in records),
        "median_updates_among_passes": (
            statistics.median(successful_updates) if successful_updates else None
        ),
        "maximum_updates_among_passes": max(successful_updates, default=None),
        "maximum_initial_load_error": max(
            record["initial_maximum_load_error"] for record in records
        ),
        "maximum_final_load_error": max(record["final_maximum_load_error"] for record in records),
        "mean_initial_per_expert_load_error": statistics.fmean(
            record["initial_mean_load_error"] for record in records
        ),
        "mean_final_per_expert_load_error": statistics.fmean(
            record["final_mean_load_error"] for record in records
        ),
        "coordinate_subgradient_failures": sum(
            record["coordinate_subgradient_failures"] for record in records
        ),
        "coordinate_interval_violations": sum(
            record["coordinate_interval_violations"] for record in records
        ),
        "nonfinite_values": sum(record["nonfinite_values"] for record in records),
        "maximum_absolute_centered_bias": max(record["maximum_absolute_bias"] for record in records),
        "mean_route_churn_per_update": statistics.fmean(churn) if churn else 0.0,
        "maximum_route_churn_per_update": max(churn, default=0.0),
    }


def reproduce_discovery(protocol):
    expected = protocol["observed_discovery"]["results"]
    mapping = {
        "source_upper_endpoint": "source_upper_endpoint",
        "interval_midpoint": "interval_midpoint",
        "source_histogram_1000": "histogram_1000",
        "histogram_10000_sensitivity": "histogram_10000",
    }
    results = {}
    for policy, expected_name in mapping.items():
        value = summarize_policy(protocol, policy, range(32), 100)
        frozen = expected[expected_name]
        for key in (
            "perfect_balance_cases",
            "cycle_cases",
            "median_updates_among_passes",
            "maximum_updates_among_passes",
        ):
            if key in frozen and value[key] != frozen[key]:
                raise RuntimeError(f"QB tie discovery did not reproduce: {expected_name}/{key}")
        results[policy] = value
    return results


def candidate_passes(result, gate):
    return (
        result["perfect_balance_cases"] == gate["perfect_balance_cases_required"]
        and result["cycle_cases"] == gate["cycles_allowed"]
        and result["unresolved_cases"] == gate["unresolved_at_50_allowed"]
        and result["coordinate_subgradient_failures"]
        == gate["coordinate_subgradient_failures_allowed"]
        and result["coordinate_interval_violations"]
        == gate["coordinate_interval_violations_allowed"]
        and result["nonfinite_values"] == gate["nonfinite_values_allowed"]
    )


def qualify(protocol_path):
    protocol_path = Path(protocol_path).expanduser().resolve()
    protocol = load_object(protocol_path)
    contract = validate_protocol(protocol)
    base = contract["base"]
    validation = contract["validation"]
    revision = repository_revision()
    discovery = reproduce_discovery(base)
    consumed = {
        policy: summarize_policy(base, policy, range(32, 96), validation["maximum_updates"])
        for policy in POLICIES
    }
    frozen_consumed = contract["consumed"]["diagnostic_convergence"]
    for policy, value in consumed.items():
        expected = frozen_consumed[policy]
        for key in (
            "perfect_balance_cases",
            "cycle_cases",
            "unresolved_cases",
            "maximum_updates_among_passes",
        ):
            if key in expected and value[key] != expected[key]:
                raise RuntimeError(f"consumed QB tie diagnostic did not reproduce: {policy}/{key}")
    unseen = {
        policy: summarize_policy(
            base, policy, range(96, 160), validation["maximum_updates"]
        )
        for policy in POLICIES
    }
    gate = validation["candidate_gate"]
    passed = candidate_passes(unseen[gate["policy"]], gate)
    return {
        "format": "speck_quantile_balancing_tie_qualification",
        "format_version": 1,
        "status": (
            "fixed_score_CPU_midpoint_qualified_training_policy_blocked"
            if passed
            else "fixed_score_CPU_tie_policy_unqualified_training_policy_blocked"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "fixed-score float64 CPU replay only; no changing-score training, hardware top-k, distributed implementation, model integration, or architecture claim",
        "protocol": {
            "path": protocol_path.relative_to(ROOT).as_posix(),
            "sha256": file_sha256(protocol_path),
        },
        "discovery_reproduction": discovery,
        "consumed_v1_holdout_reproduction": consumed,
        "unseen_validation": unseen,
        "candidate_gate": {**gate, "passed": passed},
        "decision": {
            "fixed_score_CPU_midpoint_reference_qualified": passed,
            "source_upper_endpoint_selected": False,
            "source_histogram_1000_selected": False,
            "training_tie_policy_selected": False,
            "hardware_topk_tie_semantics_qualified": False,
            "changing_score_interaction_qualified": False,
            "model_integration_authorized": False,
            "training_authorized": False,
            "promotion_authority": False,
            "next_action": "retain midpoint as a CPU tie-safe oracle; compare source upper, source histogram, and midpoint only after hardware top-k and training-parent gates close",
        },
        "runner_revision": revision,
        "runner_sha256": file_sha256(__file__),
    }


def main(argv=None):
    args = arguments(argv)
    result = qualify(args.protocol)
    atomic_json(args.output, result)
    midpoint = result["unseen_validation"]["interval_midpoint"]
    print(
        "QB finite-batch ties: "
        f"{result['status']} ({midpoint['perfect_balance_cases']}/{midpoint['cases']} "
        "unseen midpoint cases balanced)"
    )


if __name__ == "__main__":
    main()
