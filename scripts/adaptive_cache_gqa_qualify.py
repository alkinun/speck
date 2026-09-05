"""Exhaustively qualify equal-group mean reduction for adaptive GQA budgets."""

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from speck.cache_budget import eviction_loss_bound, retained_mass
from speck.cache_budget_gqa import (
    adaptive_gqa_head_budgets,
    grouped_query_eviction_loss_bound,
    grouped_query_retained_mass,
    grouped_query_selected_mass,
    reduce_grouped_query_salience,
)

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
        raise ValueError("adaptive GQA qualification requires a clean repository")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def feasible_budgets(heads, tokens, total):
    return (
        values
        for values in itertools.product(range(tokens + 1), repeat=heads)
        if sum(values) == total
    )


def normalized_random_tensor(physical_heads, group_size, tokens, seed):
    generator = random.Random(seed)
    tensor = []
    for _ in range(physical_heads):
        group = []
        for _ in range(group_size):
            row = [generator.randrange(8) for _ in range(tokens)]
            if not any(row):
                row[0] = 1
            total = sum(row)
            group.append([value / total for value in row])
        tensor.append(group)
    return tensor


def verify_source_pins(protocol):
    for source in protocol["sources"]:
        for path_key, hash_key in (
            ("local_note", "local_note_sha256"),
            ("audit", "sha256"),
            ("protocol", "protocol_sha256"),
            ("qualification", "qualification_sha256"),
        ):
            if path_key in source and file_sha256(ROOT / source[path_key]) != source[hash_key]:
                raise ValueError(f"adaptive GQA source pin changed: {path_key}")


def qualify(protocol_path):
    protocol_path = Path(protocol_path).expanduser().resolve()
    protocol = load_object(protocol_path)
    if (
        protocol.get("format") != "speck_adaptive_cache_gqa_protocol"
        or protocol.get("format_version") != 1
        or protocol.get("status") != "frozen_before_local_implementation"
    ):
        raise ValueError("adaptive GQA protocol identity is invalid")
    verify_source_pins(protocol)

    matrix_cases = 0
    budget_cases = 0
    oracle_comparisons = 0
    speck_geometry_budget_cases = 0
    maximum_reduction_gap = 0.0
    maximum_optimality_gap = 0.0
    maximum_bound_gap = 0.0
    for physical_heads in protocol["qualification_cases"]["physical_head_counts"]:
        for group_size in protocol["qualification_cases"]["equal_query_heads_per_group"]:
            for tokens in protocol["qualification_cases"]["tokens_per_head"]:
                for seed in range(protocol["qualification_cases"]["random_seeds_per_shape"]):
                    case_seed = 100_000 * physical_heads + 1_000 * group_size + 10 * tokens + seed
                    tensor = normalized_random_tensor(
                        physical_heads, group_size, tokens, case_seed
                    )
                    matrix_cases += 1
                    mean = reduce_grouped_query_salience(tensor, "mean")
                    summed = reduce_grouped_query_salience(tensor, "sum")
                    maximum_reduction_gap = max(
                        maximum_reduction_gap,
                        max(
                            abs(group_size * mean[head][token] - summed[head][token])
                            for head in range(physical_heads)
                            for token in range(tokens)
                        ),
                    )
                    previous_bound = math.inf
                    for budget in range(physical_heads * tokens + 1):
                        mean_result = adaptive_gqa_head_budgets(tensor, budget, "mean")
                        sum_result = adaptive_gqa_head_budgets(tensor, budget, "sum")
                        if (
                            mean_result["head_budgets"] != sum_result["head_budgets"]
                            or mean_result["selected"] != sum_result["selected"]
                        ):
                            raise RuntimeError("mean and sum GQA reductions changed allocation identity")
                        candidate_mass = grouped_query_retained_mass(
                            tensor, mean_result["head_budgets"]
                        )
                        oracle = -math.inf
                        for allocation in feasible_budgets(physical_heads, tokens, budget):
                            oracle = max(oracle, retained_mass(summed, allocation))
                            oracle_comparisons += 1
                        maximum_optimality_gap = max(
                            maximum_optimality_gap, oracle - candidate_mass
                        )
                        direct_bound = grouped_query_eviction_loss_bound(
                            tensor, mean_result["head_budgets"], 1.0
                        )
                        physical_bound = eviction_loss_bound(
                            mean, mean_result["head_budgets"], 1.0
                        )
                        maximum_bound_gap = max(
                            maximum_bound_gap,
                            abs(direct_bound - group_size * physical_bound),
                        )
                        if direct_bound > previous_bound + 1e-12:
                            raise RuntimeError("query-level GQA bound increased with budget")
                        previous_bound = direct_bound
                        if sum(mean_result["head_budgets"]) != budget:
                            raise RuntimeError("physical GQA allocation did not conserve capacity")
                        if physical_heads == 3 and group_size == 4:
                            speck_geometry_budget_cases += 1
                        budget_cases += 1

    counterexample = protocol["max_counterexample"]
    mean_result = adaptive_gqa_head_budgets(
        counterexample["query_salience"], counterexample["budget"], "mean"
    )
    max_result = adaptive_gqa_head_budgets(
        counterexample["query_salience"], counterexample["budget"], "max"
    )
    mean_mass = grouped_query_selected_mass(
        counterexample["query_salience"], mean_result["selected_by_head"]
    )
    max_mass = grouped_query_selected_mass(
        counterexample["query_salience"], max_result["selected_by_head"]
    )
    expected_mean = tuple(counterexample["mean_selection"])
    expected_max = tuple(counterexample["max_selection"])
    if (
        mean_result["selected"] != (expected_mean,)
        or max_result["selected"] != (expected_max,)
        or not math.isclose(mean_mass, counterexample["mean_query_retained_mass"], abs_tol=1e-12)
        or not math.isclose(max_mass, counterexample["max_query_retained_mass"], abs_tol=1e-12)
        or not mean_mass > max_mass
    ):
        raise RuntimeError("max aggregation counterexample did not reproduce")

    adaptive = (4, 1, 1)
    average_capacity = 2
    floor_ratio = 0.2
    floor_capacity = int(average_capacity * floor_ratio)
    rounded = tuple(
        round(value * (1 - floor_ratio) + floor_capacity) for value in adaptive
    )
    deficit = sum(adaptive) - sum(rounded)
    if rounded != (3, 1, 1) or deficit != 1:
        raise RuntimeError("safeguard conservation witness did not reproduce")

    if (
        maximum_reduction_gap > 1e-12
        or maximum_optimality_gap > 1e-12
        or maximum_bound_gap > 1e-12
    ):
        raise RuntimeError("equal-group mean GQA reduction failed its frozen gates")
    revision = repository_revision()
    return {
        "format": "speck_adaptive_cache_gqa_qualification",
        "format_version": 1,
        "status": "equal_group_mean_reference_qualified",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "path": protocol_path.relative_to(ROOT).as_posix(),
            "sha256": file_sha256(protocol_path),
        },
        "scope": "equal-size grouped-query reduction onto physical KV heads only; no model integration, observation-window policy, safeguard, training, novelty, or architecture claim",
        "cases": {
            "random_tensor_cases": matrix_cases,
            "random_budget_cases": budget_cases,
            "speck_gqa3_budget_cases": speck_geometry_budget_cases,
            "oracle_allocation_comparisons": oracle_comparisons,
            "mean_sum_identity_pass": True,
            "physical_capacity_conservation_pass": True,
            "query_mass_optimality_pass": True,
            "bound_scaling_and_monotonicity_pass": True,
            "maximum_mean_times_group_minus_sum_gap": maximum_reduction_gap,
            "maximum_optimality_gap": maximum_optimality_gap,
            "maximum_direct_minus_scaled_bound_gap": maximum_bound_gap,
        },
        "max_negative_control": {
            "mean_selected": mean_result["selected"],
            "max_selected": max_result["selected"],
            "mean_query_retained_mass": mean_mass,
            "max_query_retained_mass": max_mass,
            "mean_advantage": mean_mass - max_mass,
            "strict_counterexample_pass": True,
        },
        "safeguard_negative_witness": {
            "adaptive_budgets": adaptive,
            "average_capacity": average_capacity,
            "floor_ratio": floor_ratio,
            "code_like_rounded_budgets": rounded,
            "capacity_deficit": deficit,
            "conservation_failure_reproduced": True,
        },
        "decision": {
            "equal_group_mean_reference_qualified": True,
            "max_aggregation_primary_authorized": False,
            "safeguard_authorized": False,
            "upstream_code_used": False,
            "upstream_code_executed": False,
            "model_integration_authorized": False,
            "training_authorized": False,
            "novelty_gate_changed": False,
        },
        "implementation": {
            "path": "speck/cache_budget_gqa.py",
            "sha256": file_sha256(ROOT / "speck/cache_budget_gqa.py"),
        },
        "tests": {
            "path": "tests/test_cache_budget_gqa.py",
            "sha256": file_sha256(ROOT / "tests/test_cache_budget_gqa.py"),
        },
        "runner_revision": revision,
        "runner_sha256": file_sha256(__file__),
    }


def main(argv=None):
    args = arguments(argv)
    result = qualify(args.protocol)
    atomic_json(args.output, result)
    print(f"adaptive cache GQA: {result['status']}")


if __name__ == "__main__":
    main()
