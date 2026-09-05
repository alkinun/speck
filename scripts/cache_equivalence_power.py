"""Power cache-equivalence v3 from the completed v2 paired case results."""

import argparse
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

from scripts.cache_equivalence_v2 import _case_endpoint
from speck.paper_baseline import file_sha256

Z_ONE_SIDED_95 = 1.6448536269514722
Z_POWER_90 = 1.2815515655446004
SOURCE_COUNT = 11
V2_FILES = {
    "control": "results/Speck-Paper1/cache-equivalence-v2-dense-control.json",
    "kda_seed42": "results/Speck-Paper1/cache-equivalence-v2-kda-seed42.json",
    "kda_seed43": "results/Speck-Paper1/cache-equivalence-v2-kda-seed43.json",
    "kda_seed44": "results/Speck-Paper1/cache-equivalence-v2-kda-seed44.json",
    "analysis": "results/Speck-Paper1/cache-equivalence-v2-analysis.json",
}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def atomic_json(path, value):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def rounded_source_balance(cases):
    return math.ceil(cases / SOURCE_COUNT) * SOURCE_COUNT


def required_cases(standard_deviation, margin, power=0.9):
    if power != 0.9:
        raise ValueError("cache-equivalence power currently freezes 90% power")
    if standard_deviation < 0 or margin <= 0:
        raise ValueError("cache-equivalence power inputs are invalid")
    if standard_deviation == 0:
        return 1
    return math.ceil(((Z_ONE_SIDED_95 + Z_POWER_90) * standard_deviation / margin) ** 2)


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _paired_values(control_cases, candidate_cases, endpoint):
    control = {case["id"]: case for case in control_cases}
    candidate = {case["id"]: case for case in candidate_cases}
    if set(control) != set(candidate):
        raise ValueError("cache-equivalence power requires paired case ids")
    return [
        _case_endpoint(candidate[case_id], endpoint) - _case_endpoint(control[case_id], endpoint)
        for case_id in sorted(control)
    ]


def _log_rms_ratios(control_cases, candidate_cases):
    control = {case["id"]: case for case in control_cases}
    candidate = {case["id"]: case for case in candidate_cases}
    if set(control) != set(candidate):
        raise ValueError("cache-equivalence power requires paired case ids")
    values = []
    for case_id in sorted(control):
        control_value = _case_endpoint(control[case_id], "relative_rms")
        candidate_value = _case_endpoint(candidate[case_id], "relative_rms")
        if min(control_value, candidate_value) <= 0:
            raise ValueError("cache-equivalence relative RMS must be positive")
        values.append(math.log(candidate_value / control_value))
    return values


def run(repository_root):
    repository_root = Path(repository_root).expanduser().resolve()
    paths = {name: repository_root / path for name, path in V2_FILES.items()}
    reports = {name: _load(path) for name, path in paths.items()}
    if reports["analysis"].get("status") != "failed":
        raise ValueError("cache-equivalence v2 analysis must remain failed")
    control = {result["prompt_tokens"]: result for result in reports["control"]["results"]}
    candidates = {
        name: {result["prompt_tokens"]: result for result in report["results"]}
        for name, report in reports.items()
        if name.startswith("kda_seed")
    }
    endpoint_margins = {
        "argmax": 0.01,
        "high_margin": 0.002,
        "js": 0.0001,
        "top10": 0.02,
        "free": 0.05,
    }
    rows = []
    maximums = {endpoint: 0 for endpoint in (*endpoint_margins, "log_relative_rms")}
    for candidate_id, by_length in candidates.items():
        for length in sorted(control):
            for endpoint, margin in endpoint_margins.items():
                paired = _paired_values(
                    control[length]["cases"], by_length[length]["cases"], endpoint
                )
                standard_deviation = statistics.stdev(paired)
                raw = required_cases(standard_deviation, margin)
                balanced = rounded_source_balance(raw)
                maximums[endpoint] = max(maximums[endpoint], balanced)
                rows.append(
                    {
                        "checkpoint_id": candidate_id,
                        "prompt_tokens": length,
                        "endpoint": endpoint,
                        "margin": margin,
                        "v2_cases": len(paired),
                        "v2_mean_paired_difference": statistics.fmean(paired),
                        "v2_paired_standard_deviation": standard_deviation,
                        "required_cases_90_power": raw,
                        "source_balanced_cases": balanced,
                    }
                )
            paired = _log_rms_ratios(control[length]["cases"], by_length[length]["cases"])
            standard_deviation = statistics.stdev(paired)
            margin = math.log(1.5)
            raw = required_cases(standard_deviation, margin)
            balanced = rounded_source_balance(raw)
            maximums["log_relative_rms"] = max(maximums["log_relative_rms"], balanced)
            rows.append(
                {
                    "checkpoint_id": candidate_id,
                    "prompt_tokens": length,
                    "endpoint": "log_relative_rms",
                    "margin": margin,
                    "v2_cases": len(paired),
                    "v2_mean_paired_difference": statistics.fmean(paired),
                    "v2_paired_standard_deviation": standard_deviation,
                    "required_cases_90_power": raw,
                    "source_balanced_cases": balanced,
                }
            )
    primary = (
        "argmax",
        "high_margin",
        "js",
        "top10",
        "log_relative_rms",
    )
    selected = max(maximums[endpoint] for endpoint in primary)
    free_cases = maximums["free"]
    comparisons_per_length = selected * 32
    zero_event_upper = 1 - 0.05 ** (1 / comparisons_per_length)
    return {
        "format": "speck_cache_equivalence_power_analysis",
        "format_version": 1,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            name: {
                "path": path.relative_to(repository_root).as_posix(),
                "sha256": file_sha256(path),
            }
            for name, path in paths.items()
        },
        "method": {
            "alpha_one_sided": 0.05,
            "power": 0.9,
            "assumed_true_paired_difference": 0.0,
            "formula": "ceil(((z_0.95 + z_0.90) * v2_paired_sd / margin)^2)",
            "variance_source": "observed v2 case-level paired differences",
            "rounding": "up to a multiple of 11 validation sources",
            "limitations": [
                "normal-approximation planning calculation, not a result interval",
                "v2 variance estimates use 33 short or 11 4K cases",
                "power is calibrated under equal true candidate/control behavior",
            ],
        },
        "rows": rows,
        "maximum_source_balanced_cases": maximums,
        "v3_primary_endpoints": list(primary),
        "selected_v3_cases_per_length": selected,
        "selected_cases_per_source_per_length": selected // SOURCE_COUNT,
        "selected_common_history_comparisons_per_length": comparisons_per_length,
        "zero_event_one_sided_95_upper_rate": zero_event_upper,
        "free_running_cases_required_if_primary": free_cases,
        "free_running_over_selected_case_ratio": free_cases / selected,
        "authority_decision": {
            "free_running": "mandatory descriptive risk endpoint without v3 pass/fail authority",
            "reason": "exact free-running identity is discontinuous after low-margin token changes, dense diverges on 15-27% of v2 cases, and 90%-powered five-point non-inferiority would require up to 1683 cases per seed/length",
            "primary": "common-history distribution, ranking, and margin-conditioned decision fidelity",
        },
        "resource_plan": {
            "checkpoint_runs": 4,
            "case_lengths_per_checkpoint": 4,
            "maximum_evaluation_batch_size": 11,
            "maximum_gpu_minutes_per_checkpoint": 5,
            "maximum_total_gpu_minutes": 20,
            "maximum_result_bytes": 30000000,
            "observed_v2_peak_allocated_bytes": max(
                reports[name]["peak_allocated_bytes"] for name in reports if name != "analysis"
            ),
        },
    }


def main(argv=None):
    args = arguments(argv)
    report = run(args.repository_root)
    atomic_json(args.output, report)
    print(
        f"selected {report['selected_v3_cases_per_length']} cases/length; "
        f"free-running primary would require {report['free_running_cases_required_if_primary']}"
    )


if __name__ == "__main__":
    main()
