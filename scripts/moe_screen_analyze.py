"""Collect a completed local MoE screen under its frozen ranking and stability rules."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("screen", type=Path)
    parser.add_argument("checkpoint_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assess_arm(name, arm, summary, analysis):
    validation = summary.get("validation_history", [])
    final = validation[-1] if validation else None
    complete = (
        summary.get("partial") is False
        and summary.get("completed_steps") == summary.get("steps")
        and final is not None
        and final.get("step") == summary.get("steps")
    )
    loss = final.get("validation_loss") if final else None
    reasons = []
    if not complete:
        reasons.append("run is incomplete or lacks final-step validation")
    if loss is None or not isinstance(loss, (int, float)) or not math.isfinite(loss):
        reasons.append("final validation loss is missing or non-finite")

    routed = arm["router_parameters"] > 0
    stability = None
    if routed:
        start = analysis["stability_window_starts_at_tokens"]
        snapshots = [
            layer
            for point in summary.get("routing_history", [])
            if point.get("training_tokens", 0) >= start
            for layer in point.get("layers", [])
        ]
        if not snapshots:
            reasons.append("routing stability window is missing")
        else:
            numeric = [
                value
                for item in snapshots
                for key, value in item.items()
                if key != "layer" and isinstance(value, (int, float))
            ]
            maximum_cv = max(item["utilization_cv"] for item in snapshots)
            maximum_zero = max(item["zero_load_experts"] for item in snapshots)
            minimum_entropy = min(item["normalized_entropy"] for item in snapshots)
            finite = all(math.isfinite(value) for value in numeric)
            limits = analysis["stability_limits"]
            if limits["all_values_finite"] and not finite:
                reasons.append("routing diagnostics contain non-finite values")
            if maximum_cv > limits["maximum_normalized_utilization_cv"]:
                reasons.append("routing utilization CV exceeds its frozen limit")
            if maximum_zero > limits["maximum_zero_load_experts"]:
                reasons.append("one or more experts receive zero load")
            if minimum_entropy < limits["minimum_normalized_entropy"]:
                reasons.append("routing entropy falls below its frozen limit")
            stability = {
                "samples": len(snapshots),
                "all_values_finite": finite,
                "maximum_normalized_utilization_cv": maximum_cv,
                "maximum_zero_load_experts": maximum_zero,
                "minimum_normalized_entropy": minimum_entropy,
                "maximum_logit_rms": max(item["logit_rms"] for item in snapshots),
                "maximum_selection_bias_rms": max(item["selection_bias_rms"] for item in snapshots),
            }
    return {
        "name": name,
        "complete": complete,
        "eligible": not reasons,
        "final_validation_loss": loss,
        "total_parameters": arm["parameters"],
        "active_parameters": arm["active_parameters"],
        "stability": stability,
        "ineligibility_reasons": reasons,
    }


def select_arm(names, results, tie_margin, preference):
    eligible = [results[name] for name in names if results[name]["eligible"]]
    if not eligible:
        return {"selected": None, "reason": "no eligible completed arms", "ranked": []}
    ranked = sorted(eligible, key=lambda item: (item["final_validation_loss"], item["name"]))
    best = ranked[0]["final_validation_loss"]
    tied = {item["name"] for item in ranked if item["final_validation_loss"] <= best + tie_margin}
    selected = next((name for name in preference if name in tied), None)
    if selected is None:
        raise ValueError("tie-break order does not cover every eligible arm")
    return {
        "selected": selected,
        "reason": "frozen tie-break order among arms inside the single-seed noise margin",
        "best_observed_loss": best,
        "tied_with_best": sorted(tied),
        "ranked": [item["name"] for item in ranked],
    }


def analyze(screen_path, checkpoint_root):
    screen_path = Path(screen_path).resolve()
    checkpoint_root = Path(checkpoint_root).resolve()
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    if screen.get("format") != "speck_moe_design_screen" or screen.get("format_version") != 2:
        raise ValueError("analysis requires a version-2 MoE design screen")
    results = {}
    for name, arm in screen["arms"].items():
        path = checkpoint_root / f"SpeckLC-150M-MoEScreen-{name}" / "run_summary.json"
        if path.is_file():
            summary = json.loads(path.read_text(encoding="utf-8"))
            result = assess_arm(name, arm, summary, screen["analysis"])
            result["run_summary"] = {"path": str(path), "sha256": file_sha256(path)}
        else:
            result = {
                "name": name,
                "complete": False,
                "eligible": False,
                "final_validation_loss": None,
                "total_parameters": arm["parameters"],
                "active_parameters": arm["active_parameters"],
                "stability": None,
                "ineligibility_reasons": ["run summary is missing"],
            }
        results[name] = result

    tie_margin = screen["analysis"]["single_seed_tie_margin_nats"]
    preferences = screen["analysis"]["tie_break_order"]
    decisions = {
        "granularity": select_arm(
            ["g8", "g16", "g32"], results, tie_margin, preferences["granularity"]
        ),
        "placement_capacity": select_arm(
            ["g8", "p-dense-first", "p-interleaved", "p-shared"],
            results,
            tie_margin,
            preferences["placement_capacity"],
        ),
        "overall_moe_design": select_arm(
            ["g8", "g16", "g32", "p-dense-first", "p-interleaved", "p-shared"],
            results,
            tie_margin,
            preferences["overall_moe_design"],
        ),
        "dense_versus_moe": {
            "selected": None,
            "reason": "explicitly outside the scope of this design screen",
        },
    }
    return {
        "format": "speck_moe_design_screen_result",
        "format_version": 1,
        "screen": {"path": str(screen_path), "sha256": file_sha256(screen_path)},
        "checkpoint_root": str(checkpoint_root),
        "arms": results,
        "decisions": decisions,
    }


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv=None):
    args = arguments(argv)
    report = analyze(args.screen / "screen.json", args.checkpoint_root)
    atomic_json(args.output, report)
    print(json.dumps(report["decisions"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
