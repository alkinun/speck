"""Summarize predeclared score deltas across four registered tail-pair variants."""

import argparse
import json
import os
from pathlib import Path

from speck.checkpoint import file_sha256

_VARIANTS = ("control-final", "constant-final", "control-average", "constant-average")
_CONTRASTS = {
    "constant_final_minus_control_final": ("control-final", "constant-final"),
    "control_average_minus_control_final": ("control-final", "control-average"),
    "constant_average_minus_constant_final": ("constant-final", "constant-average"),
    "constant_average_minus_control_average": ("control-average", "constant-average"),
}
_BANANA_METRICS = ("accuracy", "weighted_accuracy", "overall_elo", "overall_elo_unrounded")
_BANANA_CATEGORY_METRICS = ("accuracy", "weighted_accuracy", "elo", "elo_unrounded")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("finalization_dir", type=Path)
    return parser.parse_args(argv)


def _load_json(path, label):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _numeric_scores(values, label):
    if not isinstance(values, dict):
        raise ValueError(f"{label} scores are missing")
    scores = {
        key: value
        for key, value in values.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if set(scores) != set(values):
        raise ValueError(f"{label} scores must all be numeric")
    if not scores:
        raise ValueError(f"{label} scores are empty")
    return scores


def _bananamind_scores(summary):
    scores = {
        key: summary[key]
        for key in _BANANA_METRICS
        if isinstance(summary.get(key), (int, float)) and not isinstance(summary.get(key), bool)
    }
    categories = summary.get("categories", {})
    if not isinstance(categories, dict):
        raise ValueError("BananaMind categories must be an object")
    for category, values in categories.items():
        if not isinstance(values, dict):
            raise ValueError("BananaMind category must be an object")
        for metric in _BANANA_CATEGORY_METRICS:
            value = values.get(metric)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                scores[f"categories/{category}/{metric}"] = value
    if not scores:
        raise ValueError("BananaMind summary has no comparable scores")
    return scores


def _benchmark_scores(record, benchmark):
    if benchmark == "open_slm":
        return _numeric_scores(record[benchmark]["scores_percent"], "Open-SLM")
    return _bananamind_scores(record[benchmark]["summary"])


def compare(finalization_dir):
    finalization_dir = Path(finalization_dir).expanduser().resolve()
    finalization_path = finalization_dir / "finalization.json"
    finalization = _load_json(finalization_path, "tail-pair finalization")
    if (
        finalization.get("format") != "speck_tail_pair_finalization"
        or finalization.get("format_version") != 1
    ):
        raise ValueError("unsupported tail-pair finalization format")
    finalization_sha256 = file_sha256(finalization_path)
    records = {}
    identities = {}
    for variant in _VARIANTS:
        path = finalization_dir / "results" / f"{variant}.json"
        record = _load_json(path, f"{variant} result")
        if (
            record.get("format") != "speck_tail_pair_result"
            or record.get("format_version") != 1
            or record.get("variant") != variant
            or record.get("finalization", {}).get("sha256") != finalization_sha256
            or Path(record.get("finalization", {}).get("path", "")).expanduser().resolve()
            != finalization_path
        ):
            raise ValueError(f"{variant} result does not match finalization")
        records[variant] = record
        identities[variant] = {"path": str(path), "sha256": file_sha256(path)}

    benchmark_sets = [
        {benchmark for benchmark in ("open_slm", "bananamind") if benchmark in record}
        for record in records.values()
    ]
    if not benchmark_sets[0] or any(current != benchmark_sets[0] for current in benchmark_sets[1:]):
        raise ValueError("registered variants must contain the same benchmark results")

    scores = {}
    for benchmark in sorted(benchmark_sets[0]):
        protocols = [record[benchmark].get("protocol") for record in records.values()]
        if any(protocol != protocols[0] for protocol in protocols[1:]):
            raise ValueError(f"{benchmark} protocols do not match")
        benchmark_scores = {
            variant: _benchmark_scores(record, benchmark) for variant, record in records.items()
        }
        metric_sets = [set(values) for values in benchmark_scores.values()]
        if any(current != metric_sets[0] for current in metric_sets[1:]):
            raise ValueError(f"{benchmark} metric sets do not match")
        scores[benchmark] = benchmark_scores

    contrasts = {}
    for name, (baseline, treatment) in _CONTRASTS.items():
        contrasts[name] = {"baseline": baseline, "treatment": treatment}
        for benchmark, benchmark_scores in scores.items():
            contrasts[name][benchmark] = {
                metric: benchmark_scores[treatment][metric] - benchmark_scores[baseline][metric]
                for metric in sorted(benchmark_scores[baseline])
            }

    comparison = {
        "format": "speck_tail_pair_comparison",
        "format_version": 1,
        "finalization": {"path": str(finalization_path), "sha256": finalization_sha256},
        "results": identities,
        "contrasts": contrasts,
    }
    output = finalization_dir / "comparison.json"
    if output.exists():
        raise FileExistsError(f"tail-pair comparison already exists: {output}")
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return comparison


def main():
    args = parse_args()
    comparison = compare(args.finalization_dir)
    print(f"Wrote {len(comparison['contrasts'])} predeclared tail-pair contrasts")


if __name__ == "__main__":
    main()
