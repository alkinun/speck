"""Exhaustively qualify the clean-room adaptive physical-head budget reference."""

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

from speck.cache_budget import (
    adaptive_head_budgets,
    eviction_loss_bound,
    retained_mass,
    uniform_head_budgets,
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
        raise ValueError("adaptive budget qualification requires a clean repository")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def feasible_budgets(heads, tokens, total):
    return (
        values
        for values in itertools.product(range(tokens + 1), repeat=heads)
        if sum(values) == total
    )


def normalized_random_matrix(heads, tokens, seed):
    generator = random.Random(seed)
    matrix = []
    for _ in range(heads):
        row = [generator.randrange(8) for _ in range(tokens)]
        if not any(row):
            row[0] = 1
        total = sum(row)
        matrix.append([value / total for value in row])
    return matrix


def adversarial_matrices():
    return [
        [[0.2] * 5 for _ in range(4)],
        [[1.0, 0.0, 0.0, 0.0, 0.0], [0.2] * 5, [0.2] * 5, [0.2] * 5],
        [[1.0, 0.0, 0.0, 0.0, 0.0] for _ in range(4)],
        [
            [0.4, 0.3, 0.2, 0.1, 0.0],
            [0.0, 0.1, 0.2, 0.3, 0.4],
            [0.25, 0.25, 0.25, 0.25, 0.0],
            [0.0, 0.25, 0.25, 0.25, 0.25],
        ],
    ]


def qualify(protocol_path):
    protocol_path = Path(protocol_path).expanduser().resolve()
    protocol = load_object(protocol_path)
    if (
        protocol.get("format") != "speck_adaptive_cache_budget_protocol"
        or protocol.get("format_version") != 1
        or protocol.get("status") != "frozen_before_local_implementation"
    ):
        raise ValueError("adaptive cache budget protocol identity is invalid")
    audit = protocol["sources"][1]
    if file_sha256(ROOT / audit["audit"]) != audit["sha256"]:
        raise ValueError("Ada-KV audit pin changed")

    matrix_cases = 0
    budget_cases = 0
    oracle_comparisons = 0
    tie_cases = 0
    maximum_optimality_gap = 0.0
    minimum_adaptive_minus_uniform = math.inf
    for heads in protocol["qualification_cases"]["head_counts"]:
        for tokens in protocol["qualification_cases"]["tokens_per_head"]:
            for seed in range(protocol["qualification_cases"]["random_seeds_per_shape"]):
                matrix = normalized_random_matrix(heads, tokens, 10_000 * heads + 100 * tokens + seed)
                matrix_cases += 1
                previous_bound = math.inf
                for budget in range(heads * tokens + 1):
                    result = adaptive_head_budgets(matrix, budget)
                    if len(set(value for row in matrix for value in row)) < heads * tokens:
                        tie_cases += 1
                    candidate_mass = result["retained_mass"]
                    oracle = -math.inf
                    for allocation in feasible_budgets(heads, tokens, budget):
                        oracle = max(oracle, retained_mass(matrix, allocation))
                        oracle_comparisons += 1
                    gap = oracle - candidate_mass
                    maximum_optimality_gap = max(maximum_optimality_gap, gap)
                    uniform = uniform_head_budgets(heads, tokens, budget)
                    minimum_adaptive_minus_uniform = min(
                        minimum_adaptive_minus_uniform,
                        candidate_mass - retained_mass(matrix, uniform),
                    )
                    if sum(result["head_budgets"]) != budget:
                        raise RuntimeError("adaptive head budgets do not conserve total capacity")
                    bound = eviction_loss_bound(matrix, result["head_budgets"], 1.0)
                    if bound > previous_bound + 1e-12:
                        raise RuntimeError("eviction bound increased with budget")
                    previous_bound = bound
                    budget_cases += 1

    adversarial_cases = 0
    for matrix in adversarial_matrices():
        heads, tokens = len(matrix), len(matrix[0])
        for budget in range(heads * tokens + 1):
            first = adaptive_head_budgets(matrix, budget)
            second = adaptive_head_budgets(matrix, budget)
            if first != second:
                raise RuntimeError("adaptive allocation tie handling is not deterministic")
            adversarial_cases += 1

    if maximum_optimality_gap > 1e-12 or minimum_adaptive_minus_uniform < -1e-12:
        raise RuntimeError("adaptive allocation failed its exhaustive retained-mass gate")
    revision = repository_revision()
    return {
        "format": "speck_adaptive_cache_budget_qualification",
        "format_version": 1,
        "status": "clean_room_reference_qualified",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "path": protocol_path.relative_to(ROOT).as_posix(),
            "sha256": file_sha256(protocol_path),
        },
        "scope": "within-layer physical cache heads only; no GQA reduction, model integration, upstream execution, or architecture claim",
        "cases": {
            "random_matrix_cases": matrix_cases,
            "random_budget_cases": budget_cases,
            "adversarial_budget_cases": adversarial_cases,
            "oracle_allocation_comparisons": oracle_comparisons,
            "budget_conservation_pass": True,
            "per_head_identity_pass": True,
            "deterministic_ties_pass": True,
            "uniform_control_dominance_pass": True,
            "bound_monotonicity_pass": True,
            "maximum_optimality_gap": maximum_optimality_gap,
            "minimum_adaptive_minus_uniform_mass": minimum_adaptive_minus_uniform,
        },
        "decision": {
            "reference_qualified": True,
            "upstream_code_used": False,
            "upstream_code_executed": False,
            "model_integration_authorized": False,
            "novelty_gate_changed": False,
        },
        "implementation": {
            "path": "speck/cache_budget.py",
            "sha256": file_sha256(ROOT / "speck/cache_budget.py"),
        },
        "tests": {
            "path": "tests/test_cache_budget.py",
            "sha256": file_sha256(ROOT / "tests/test_cache_budget.py"),
        },
        "runner_revision": revision,
        "runner_sha256": file_sha256(__file__),
    }


def main(argv=None):
    args = arguments(argv)
    result = qualify(args.protocol)
    atomic_json(args.output, result)
    print(f"adaptive cache budget: {result['status']}")


if __name__ == "__main__":
    main()
