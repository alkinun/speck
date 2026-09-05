"""Exhaustively qualify conservation-safe adaptive cache safeguards."""

import argparse
import hashlib
import itertools
import json
import os
import subprocess
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from speck.cache_budget import uniform_head_budgets
from speck.cache_budget_safeguard import safeguard_apportionment

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
        raise ValueError("adaptive safeguard qualification requires a clean repository")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def verify_source_pins(protocol):
    for source in protocol["sources"]:
        for path_key, hash_key in (
            ("local_note", "local_note_sha256"),
            ("audit", "sha256"),
            ("protocol", "protocol_sha256"),
            ("qualification", "qualification_sha256"),
        ):
            if path_key in source and file_sha256(ROOT / source[path_key]) != source[hash_key]:
                raise ValueError(f"adaptive safeguard source pin changed: {path_key}")


def deviation(allocation, targets):
    differences = [abs(Fraction(value) - target) for value, target in zip(allocation, targets)]
    return sum(differences), sum(difference * difference for difference in differences)


def oracle_key(allocation, targets):
    l1, squared_l2 = deviation(allocation, targets)
    return l1, squared_l2, tuple(-value for value in allocation)


def qualify(protocol_path):
    protocol_path = Path(protocol_path).expanduser().resolve()
    protocol = load_object(protocol_path)
    if (
        protocol.get("format") != "speck_adaptive_cache_safeguard_protocol"
        or protocol.get("format_version") != 1
        or protocol.get("status") != "frozen_before_local_implementation"
    ):
        raise ValueError("adaptive safeguard protocol identity is invalid")
    verify_source_pins(protocol)

    cases = 0
    oracle_comparisons = 0
    endpoint_cases = 0
    lower_floor_cases = 0
    maximum_conservation_error = 0
    for heads in protocol["qualification_cases"]["head_counts"]:
        for tokens in protocol["qualification_cases"]["tokens_per_head"]:
            allocations = list(itertools.product(range(tokens + 1), repeat=heads))
            by_total = {}
            for allocation in allocations:
                by_total.setdefault(sum(allocation), []).append(allocation)
            for adaptive in allocations:
                total = sum(adaptive)
                feasible = by_total[total]
                for raw_fraction in protocol["qualification_cases"]["uniform_fractions"]:
                    result = safeguard_apportionment(adaptive, tokens, raw_fraction)
                    fraction = result["uniform_fraction"]
                    budgets = result["head_budgets"]
                    targets = result["targets"]
                    oracle = min(feasible, key=lambda value: oracle_key(value, targets))
                    oracle_comparisons += len(feasible)
                    if budgets != oracle:
                        raise RuntimeError("safeguard did not match the exact apportionment oracle")
                    if any(
                        budget not in {target.numerator // target.denominator, -(-target.numerator // target.denominator)}
                        for budget, target in zip(budgets, targets)
                    ):
                        raise RuntimeError("safeguard budget is not a target floor or ceiling")
                    lower_floor = (fraction * Fraction(total, heads)).numerator // (
                        fraction * Fraction(total, heads)
                    ).denominator
                    if any(budget < lower_floor for budget in budgets):
                        raise RuntimeError("safeguard violated the uniform floor guarantee")
                    maximum_conservation_error = max(
                        maximum_conservation_error, abs(sum(budgets) - total)
                    )
                    lower_floor_cases += 1
                    if fraction == 0:
                        if budgets != adaptive:
                            raise RuntimeError("zero safeguard endpoint changed adaptive budgets")
                        endpoint_cases += 1
                    if fraction == 1:
                        if budgets != uniform_head_budgets(heads, tokens, total):
                            raise RuntimeError("full safeguard endpoint changed the uniform control")
                        endpoint_cases += 1
                    cases += 1

    witness = protocol["negative_control"]
    adaptive = tuple(witness["adaptive_budgets"])
    code_like = tuple(
        round(
            value * (1 - float(Fraction(witness["uniform_fraction"])))
            + int((sum(adaptive) / len(adaptive)) * float(Fraction(witness["uniform_fraction"])))
        )
        for value in adaptive
    )
    reference = safeguard_apportionment(
        adaptive, witness["token_capacity"], witness["uniform_fraction"]
    )["head_budgets"]
    if (
        code_like != tuple(witness["code_like_independent_rounding"])
        or sum(code_like) != witness["code_like_total"]
        or reference != tuple(witness["reference_result"])
        or sum(reference) != witness["required_total"]
    ):
        raise RuntimeError("safeguard conservation witness did not reproduce")

    revision = repository_revision()
    return {
        "format": "speck_adaptive_cache_safeguard_qualification",
        "format_version": 1,
        "status": "conservation_safe_reference_qualified",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "path": protocol_path.relative_to(ROOT).as_posix(),
            "sha256": file_sha256(protocol_path),
        },
        "scope": "exact integer apportionment control only; no salience acquisition, model integration, training, quality benefit, novelty, or architecture claim",
        "cases": {
            "apportionment_cases": cases,
            "oracle_allocation_comparisons": oracle_comparisons,
            "endpoint_cases": endpoint_cases,
            "uniform_lower_floor_cases": lower_floor_cases,
            "capacity_and_conservation_pass": maximum_conservation_error == 0,
            "floor_or_ceiling_pass": True,
            "exact_l1_l2_oracle_pass": True,
            "deterministic_tie_pass": True,
            "endpoint_pass": True,
            "uniform_lower_floor_pass": True,
            "maximum_absolute_conservation_error": maximum_conservation_error,
        },
        "negative_control": {
            "adaptive_budgets": adaptive,
            "code_like_rounded_budgets": code_like,
            "code_like_total": sum(code_like),
            "reference_budgets": reference,
            "reference_total": sum(reference),
            "conservation_failure_preserved": True,
        },
        "decision": {
            "conservation_safe_reference_qualified": True,
            "paper_alpha_semantics_selected": False,
            "upstream_behavior_reproduction_claimed": False,
            "primary_safeguard_authorized": False,
            "upstream_code_used": False,
            "upstream_code_executed": False,
            "model_integration_authorized": False,
            "training_authorized": False,
            "quality_benefit_claimed": False,
            "novelty_gate_changed": False,
        },
        "implementation": {
            "path": "speck/cache_budget_safeguard.py",
            "sha256": file_sha256(ROOT / "speck/cache_budget_safeguard.py"),
        },
        "tests": {
            "path": "tests/test_cache_budget_safeguard.py",
            "sha256": file_sha256(ROOT / "tests/test_cache_budget_safeguard.py"),
        },
        "runner_revision": revision,
        "runner_sha256": file_sha256(__file__),
    }


def main(argv=None):
    args = arguments(argv)
    result = qualify(args.protocol)
    atomic_json(args.output, result)
    print(f"adaptive cache safeguard: {result['status']}")


if __name__ == "__main__":
    main()
