"""Summarize repeated throughput receipts without mixing benchmark geometries."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from speck.provenance.io import file_sha256


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    mean = statistics.mean(ordered)
    median = statistics.median(ordered)
    deviation = statistics.stdev(ordered) if len(ordered) > 1 else 0.0
    return {
        "count": len(ordered),
        "mean": mean,
        "median": median,
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "standard_deviation": deviation,
        "coefficient_of_variation": deviation / mean if mean else None,
        "maximum_absolute_relative_to_median": (
            max(abs(value - median) / median for value in ordered) if median else None
        ),
    }


def summarize(paths: list[str | Path]) -> dict:
    """Validate and summarize receipts from repeated equivalent benchmark runs."""

    resolved = [Path(path).resolve() for path in paths]
    if len(resolved) < 2:
        raise ValueError("provide at least two receipts to estimate run-to-run noise")
    if len(set(resolved)) != len(resolved):
        raise ValueError("receipt paths must be distinct")

    records = []
    for path in resolved:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("format") != "speck_throughput_probe":
            raise ValueError(f"{path}: unsupported benchmark receipt format")
        for section in ("benchmark", "geometry", "performance", "model", "experiment"):
            if not isinstance(record.get(section), dict):
                raise ValueError(f"{path}: missing benchmark section {section}")
        records.append((path, record))

    reference = records[0][1]
    identity = {
        "experiment_fingerprint": reference["experiment"].get("fingerprint"),
        "geometry": reference["geometry"],
        "benchmark": {
            key: reference["benchmark"].get(key)
            for key in (
                "mode",
                "compiled",
                "compile_mode",
                "loss_backend",
                "activation_checkpointing",
                "deterministic",
                "training_output",
                "optimizer",
                "optimizer_step_compiled",
            )
        },
    }
    for path, record in records[1:]:
        candidate = {
            "experiment_fingerprint": record["experiment"].get("fingerprint"),
            "geometry": record["geometry"],
            "benchmark": {key: record["benchmark"].get(key) for key in identity["benchmark"]},
        }
        if candidate != identity:
            raise ValueError(f"{path}: receipt does not match the first run's configuration")

    rates = [float(record["performance"]["tokens_per_second"]) for _, record in records]
    step_seconds = [float(record["performance"]["step_seconds_median"]) for _, record in records]
    mfu = [
        float(record["performance"]["model_flops_utilization"])
        for _, record in records
        if record["performance"].get("model_flops_utilization") is not None
    ]
    mfu_summary = _summary(mfu) if len(mfu) == len(records) else None
    summary = {
        "tokens_per_second": _summary(rates),
        "step_seconds_median": _summary(step_seconds),
        "model_flops_utilization": mfu_summary,
    }
    return {
        "format": "speck_throughput_repeat_summary",
        "format_version": 1,
        "inputs": [{"path": str(path), "sha256": file_sha256(path)} for path, _ in records],
        "runs": len(records),
        "identity": identity,
        "summary": summary,
        "interpretation": (
            "All inputs share the same experiment fingerprint, geometry, and runtime settings. "
            "The maximum relative deviation from the median is a run-to-run noise band; it is "
            "not a confidence interval and does not establish a hardware-independent result."
        ),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    result = json.dumps(summarize(args.receipts), indent=2, sort_keys=True)
    print(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
