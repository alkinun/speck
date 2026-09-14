"""Reconcile completed tokenizer screens with measured, all-attempt GPU accounting."""

import json
import math
from pathlib import Path

from speck.provenance.io import file_sha256
from speck.tokenization.pilot import (
    VIEWS,
    _macro,
    _paired_document_deltas,
    _validate_run,
    validate_pilot_plan,
)


def _hours(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be finite nonnegative GPU-hours")
    return value


def validate_screen_plan(plan):
    """Check v10's execution envelope and its unchanged inherited analysis rules."""

    if (
        not isinstance(plan, dict)
        or plan.get("format_version") != 10
        or plan.get("status") != "throughput_preflight_pass_run_materializer_pending_D5_unopened"
        or plan.get("stopping", {}).get("fixed_document_mistral_tokens") != 1_200_007_273
        or plan.get("screen", {}).get("custom_rule")
        != "lowest fixed-document equal-category macro BPB; tie by lower active seconds, then smaller vocabulary, then ID"
    ):
        raise ValueError("tokenizer screen plan differs from the frozen contract")
    # Only v10's version/status and whole-document overshoot differ for this
    # validation. Keep the original plan and frozen seven-run analyzer untouched.
    validate_pilot_plan(
        {
            **plan,
            "format_version": 1,
            "status": "analysis_fixture_ready_real_runs_blocked",
            "stopping": {**plan["stopping"], "fixed_document_mistral_tokens": 1_200_000_000},
        }
    )
    return plan


def analyze_tokenizer_screen(plan, nominations, runs, accounting):
    """Apply the frozen screen ranking and cost gate; never issue launch authority.

    Run active time already includes in-loop evaluation and prior checkpoint saves.
    All-attempt spending also covers startup, final saving, and lost/replayed work.
    Confirmation estimates are full per-run costs, not overhead added to active time.
    """

    validate_screen_plan(plan)
    if (
        nominations.get("format") != "speck_tokenizer_static_nomination"
        or nominations.get("status")
        != "real_static_endpoints_nominated_for_lm_pilot_no_selection_authority"
        or nominations.get("selection_authority") is not False
    ):
        raise ValueError("screen requires the v10 plan and real static nomination")
    entries = nominations.get("nominations", [])
    if (
        len(entries) != 2
        or {entry.get("role") for entry in entries} != {"compression_endpoint", "compact_endpoint"}
        or any(not isinstance(entry.get("id"), str) or not entry["id"] for entry in entries)
    ):
        raise ValueError("screen requires two distinct nominated endpoints")
    custom_ids = {entry["id"] for entry in entries}
    baseline_id = plan["primary_baseline"]
    expected_ids = custom_ids | {baseline_id}
    if len(expected_ids) != 3:
        raise ValueError("screen requires two distinct nominated endpoints")
    runs = [_validate_run(run, plan) for run in runs]
    by_id = {run["tokenizer_id"]: run for run in runs}
    if len(runs) != 3 or set(by_id) != expected_ids or any(run["seed"] != 42 for run in runs):
        raise ValueError("screen requires exactly the three complete seed-42 runs")
    baseline = by_id[baseline_id]
    target = baseline["fixed_document"]["analytic_flops"]
    if (
        len({run["document_stream_sha256"] for run in runs}) != 1
        or len({run["backbone_manifest_sha256"] for run in runs}) != 1
        or baseline["fixed_document"]["tokenizer_tokens"]
        != plan["stopping"]["fixed_document_mistral_tokens"]
        or any(
            run["fixed_flop"]["analytic_flops"] != target
            or any(run[view]["target_analytic_flops"] != target for view in VIEWS)
            for run in runs
        )
    ):
        raise ValueError("screen has unmatched backbone, document stream, or FLOP target")
    for run in runs:
        for view in VIEWS:
            _paired_document_deltas([run], [baseline], view)
        # The two endpoint views must also evaluate the same documents.
        _paired_document_deltas(
            [run], [{**run, "fixed_document": run["fixed_flop"]}], "fixed_document"
        )
    selected = min(
        (by_id[key] for key in custom_ids),
        key=lambda run: (
            _macro(run, "fixed_document"),
            run["fixed_document"]["active_seconds"],
            run["vocab_size"],
            run["tokenizer_id"],
        ),
    )["tokenizer_id"]
    if (
        accounting.get("format") != "speck_tokenizer_pilot_screen_accounting"
        or accounting.get("format_version") != 1
        or not isinstance(accounting.get("complete"), bool)
    ):
        raise ValueError("invalid screen accounting record")
    budget = None
    status = "accounting_incomplete_confirmation_blocked"
    # This is a lower bound on the frozen projection, never an all-attempt total.
    # Missing setup, replay, final publication, and preflight costs are nonnegative:
    # they can rule out confirmation but cannot authorize it.
    active_hours = {
        key: max(run[view]["active_seconds"] for view in VIEWS) / 3600 for key, run in by_id.items()
    }
    minimum_spent = sum(active_hours.values())
    minimum_remaining = 2 * (active_hours[baseline_id] + active_hours[selected])
    lower_bound = {
        "completed_screen_active_gpu_hours": minimum_spent,
        "four_confirmation_active_gpu_hours": minimum_remaining,
        "projected_total_gpu_hours": minimum_spent + minimum_remaining,
        "ceiling_gpu_hours": plan["gpu_hour_ceiling"],
        "exceeds_ceiling": minimum_spent + minimum_remaining > plan["gpu_hour_ceiling"],
        "all_attempt_spending_complete": accounting["complete"],
        "boundary": "Lower bound on the contract's measured-cost projection, not a complete expenditure ledger or a guaranteed future runtime. Missing costs are not zero. Only an excess can determine the budget stop; a lower bound within budget never authorizes confirmation.",
    }
    if not accounting["complete"] and lower_bound["exceeds_ceiling"]:
        status = "projection_lower_bound_exceeded_retain_mistral_D5_unopened"
    if accounting["complete"]:
        costs = accounting.get("runs", [])
        by_cost = {entry["tokenizer_id"]: entry for entry in costs}
        if len(costs) != 3 or set(by_cost) != expected_ids:
            raise ValueError("accounting must cover exactly the three screen runs")
        other = _hours(accounting.get("other_spent_gpu_hours"), "other spending")
        for key, cost in by_cost.items():
            minimum = max(by_id[key][view]["active_seconds"] for view in VIEWS) / 3600
            for field in ("spent_gpu_hours", "confirmation_run_gpu_hours"):
                if _hours(cost.get(field), field) < minimum:
                    raise ValueError(f"{key} {field} omits measured active time")
        spent = other + sum(entry["spent_gpu_hours"] for entry in costs)
        remaining = 2 * (
            by_cost[baseline_id]["confirmation_run_gpu_hours"]
            + by_cost[selected]["confirmation_run_gpu_hours"]
        )
        total = spent + remaining
        fits = total <= plan["gpu_hour_ceiling"]
        budget = {
            "spent_gpu_hours": spent,
            "projected_four_confirmation_gpu_hours": remaining,
            "projected_total_gpu_hours": total,
            "ceiling_gpu_hours": plan["gpu_hour_ceiling"],
            "headroom_gpu_hours": plan["gpu_hour_ceiling"] - total,
            "fits": fits,
        }
        status = (
            "measured_budget_pass_confirmation_manifests_pending"
            if fits
            else "measured_budget_exceeded_retain_mistral_D5_unopened"
        )
    return {
        "format": "speck_tokenizer_pilot_screen_analysis",
        "format_version": 1,
        "status": status,
        "screen": {
            "seed": 42,
            "selected_custom": selected,
            "scores": {
                key: {
                    "fixed_document_macro_bpb": _macro(by_id[key], "fixed_document"),
                    "fixed_document_active_seconds": by_id[key]["fixed_document"]["active_seconds"],
                    "measured_run_active_gpu_hours": max(
                        by_id[key][view]["active_seconds"] for view in VIEWS
                    )
                    / 3600,
                }
                for key in sorted(by_id)
            },
        },
        "budget": budget,
        "projection_lower_bound": lower_bound,
        "fallback": baseline_id,
        "authority": {
            "confirmation_execution": False,
            "D5_opening": False,
            "final_selection": False,
            "flagship_training": False,
        },
    }


def load_screen_report(manifest_path):
    """Verify pinned plan, nominations, summaries, results, and accounting evidence."""

    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    identities = []

    def load_identity(identity, base=root):
        if not isinstance(identity, dict) or set(identity) != {"path", "sha256"}:
            raise ValueError("screen inputs require path and sha256")
        path = Path(identity["path"])
        path = (base / path).resolve()
        if not path.is_file() or file_sha256(path) != identity["sha256"]:
            raise ValueError(f"screen input identity mismatch: {path}")
        identities.append({"path": str(path), "sha256": identity["sha256"]})
        return path

    if (
        manifest.get("format") != "speck_tokenizer_pilot_screen_review"
        or manifest.get("format_version") != 1
    ):
        raise ValueError("invalid screen review manifest")
    plan = json.loads(load_identity(manifest["plan"]).read_text())
    nominations = json.loads(load_identity(manifest["nominations"]).read_text())
    accounting_path = load_identity(manifest["accounting"])
    accounting = json.loads(accounting_path.read_text())
    if accounting.get("complete") is True and not accounting.get("evidence"):
        raise ValueError("complete accounting requires measured timing evidence")
    for identity in accounting.get("evidence", []):
        load_identity(identity, accounting_path.parent)
    runs = []
    for identity in manifest["summaries"]:
        path = load_identity(identity)
        summary = json.loads(path.read_text())
        if (
            summary.get("format") != "speck_tokenizer_pilot_run_summary"
            or summary.get("format_version") not in (1, 2)
            or summary.get("status") != "complete"
            or summary.get("D5_opening") is not False
        ):
            raise ValueError("screen summary must be complete with D5 unopened")
        run = json.loads(load_identity(summary["result"], path.parent).read_text())
        if summary["format_version"] != run["format_version"] or summary.get(
            "missing_measurements"
        ) != run.get("missing_measurements"):
            raise ValueError("screen summary and result measurement schemas disagree")
        if (
            summary.get("run_id") != f"tokenizer-pilot-{run['tokenizer_id']}-seed-{run['seed']}"
            or summary.get("active_seconds") != run["fixed_flop"]["active_seconds"]
        ):
            raise ValueError("screen summary and result disagree")
        runs.append(run)
    result = analyze_tokenizer_screen(plan, nominations, runs, accounting)
    result["inputs"] = {
        "review": {"path": str(manifest_path), "sha256": file_sha256(manifest_path)},
        "artifacts": identities,
    }
    return result
