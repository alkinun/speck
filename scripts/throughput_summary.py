"""Summarize repeated throughput receipts without mixing benchmark geometries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


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

    records, inputs, identities = [], [], []
    seen = set()
    for path in resolved:
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest in seen:
            raise ValueError("duplicate receipt content is not an independent repeat")
        seen.add(digest)
        record = json.loads(payload)
        if record.get("format") != "speck_throughput_probe" or record.get("format_version") != 2:
            raise ValueError(f"{path}: require a version 2 benchmark receipt")
        try:
            benchmark = record["benchmark"]
            identity = {
                "benchmark": {
                    key: benchmark[key]
                    for key in (
                        "mode",
                        "compiled",
                        "compile_mode",
                        "aggressive_fusion",
                        "loss_backend",
                        "activation_checkpointing",
                        "deterministic",
                        "training_output",
                        "optimizer",
                        "optimizer_step_compiled",
                        "seed",
                        "steps",
                        "warmup_steps",
                        "peak_tflops",
                    )
                },
                "geometry": record["geometry"],
                "model": record["model"],
                "experiment": {
                    key: record["experiment"][key] for key in ("fingerprint", "manifest")
                },
                "environment": {
                    key: record["environment"][key]
                    for key in (
                        "device",
                        "device_name",
                        "device_capability",
                        "torch",
                        "cuda",
                        "packages",
                        "cublas_workspace_config",
                        "git_revision",
                        "git_dirty",
                    )
                },
            }
            identity["environment"]["kernel_caches"] = record["environment"].get("kernel_caches")
            if (
                not identity["experiment"]["fingerprint"]
                or not identity["environment"]["git_revision"]
            ):
                raise ValueError("missing configuration or source identity")
            if identity["environment"]["git_dirty"] is not False:
                raise ValueError("repeat comparisons require a clean source revision")
            if record["quality"]["stable"] is not True:
                raise ValueError("unstable receipt: diagnose before summarizing repeats")
            for key in ("tokens_per_second", "step_seconds_median"):
                value = record["performance"][key]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError(f"{key} must be finite and positive")
            mfu = record["performance"]["model_flops_utilization"]
            peak = benchmark["peak_tflops"]
            if peak is not None:
                if (
                    isinstance(peak, bool)
                    or not isinstance(peak, (int, float))
                    or not math.isfinite(peak)
                    or peak <= 0
                ):
                    raise ValueError("peak_tflops must be finite and positive")
                expected = (
                    record["model"]["flops_per_token"]
                    * record["performance"]["tokens_per_second"]
                    / (peak * 1e12)
                )
                if (
                    isinstance(mfu, bool)
                    or not isinstance(mfu, (int, float))
                    or not math.isfinite(mfu)
                    or not math.isclose(mfu, expected)
                ):
                    raise ValueError("MFU does not match the recorded FLOP convention and peak")
            elif mfu is not None:
                raise ValueError("MFU requires a declared peak_tflops")
        except (KeyError, TypeError) as error:
            raise ValueError(f"{path}: incomplete benchmark identity or measurements") from error
        identities.append(identity)
        records.append((path, record))
        inputs.append({"path": str(path), "sha256": digest})

    identity = identities[0]
    if any(candidate != identity for candidate in identities[1:]):
        raise ValueError("receipt does not match the first run's hardware, source or configuration")

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
        "inputs": inputs,
        "runs": len(records),
        "identity": identity,
        "summary": summary,
        "interpretation": (
            "All inputs share recorded hardware, source, software, geometry and runtime settings. "
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
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(result + "\n")


if __name__ == "__main__":
    main()
