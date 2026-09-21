"""Validate the cross-stage working plan without authorizing any run.

This check catches arithmetic and receipt drift between the numeric plan and the
design-only study packets. It never acquires data, selects sources, or launches training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def validate(plan_path: str | Path = ROOT / "experiments/main-data/plan.json") -> dict:
    """Reconcile working ceilings and local receipts, not scientific readiness.

    An alternate plan path supports offline regression tests; packet and receipt
    paths still resolve against the repository root, never the caller's cwd.
    """
    plan = json.loads(Path(plan_path).resolve().read_text())
    compute = plan["compute"]
    reservations = compute["reservations_gpu_hours"]
    if sum(reservations.values()) != compute["requested_total_gpu_hours"]:
        raise ValueError("compute reservations do not sum to the requested allocation")

    breakdown = compute["data_experiments_breakdown_gpu_hours"]
    if sum(breakdown.values()) != reservations["data_experiments"]:
        raise ValueError("data-experiment breakdown does not match its reservation")

    protected = compute["proposed_protected_breakdown_gpu_hours"]
    if sum(protected.values()) != reservations["protected_recovery_and_evaluation"]:
        raise ValueError("protected breakdown does not match its reservation")
    # The Slurm wave validator carries its own scheduled/protected constants and
    # rejects any execution plan that disagrees with them. Bind them here so a
    # revised reservation table cannot pass this check and then fail at wave
    # submission on the cluster, which is the worst place to discover the drift.
    from speck.operations import slurm

    protected_hours = reservations["protected_recovery_and_evaluation"]
    scheduled = compute["requested_total_gpu_hours"] - protected_hours
    if (
        slurm.TOTAL_GPU_HOURS != compute["requested_total_gpu_hours"]
        or slurm.MANDATORY_GPU_HOURS != scheduled
        or slurm.RESERVE_GPU_HOURS != protected_hours
    ):
        raise ValueError("speck.operations.slurm budget constants drift from the reservation table")

    first_scenario = compute["main_scenarios"][0]
    if first_scenario["tokens"] != plan["main_pretraining"]["target_tokens"]:
        raise ValueError("first compute scenario does not match the working token horizon")
    if compute["main_pretraining_reference_gpu_hours"] != first_scenario["reference_gpu_hours"]:
        raise ValueError("main reference GPU-hours drift from the first compute scenario")

    # Eligible preparation includes selection headroom. It is not extra exposure
    # and must not be charged as additional production training tokens.
    mixture = plan["main_pretraining"]["mixture"]
    if sum(item["weight_percent"] for item in mixture) != 100:
        raise ValueError("main mixture weights must sum to 100 percent")
    target = plan["main_pretraining"]["target_tokens"]
    if sum(item["exposure_tokens"] for item in mixture) != target:
        raise ValueError("main mixture exposure does not match the working horizon")
    unique_target = plan["main_pretraining"]["target_unique_eligible_tokens"]
    if sum(item["eligible_unique_token_preparation_target"] for item in mixture) != unique_target:
        raise ValueError("main mixture preparation targets do not match the eligible-token target")

    # Current evidence indexes repeat the working horizon so that a stale
    # readiness receipt cannot silently describe the superseded 100B target.
    readiness = _load("experiments/main-data/source-readiness.json")
    readiness_target = readiness["horizon_accounting"]["working_target"]
    if (
        readiness_target["total_exposure_tokens"] != target
        or readiness_target["minimum_unique_preparation_tokens"] != unique_target
    ):
        raise ValueError("source-readiness horizon drifts from the working plan")

    code_expansion = _load("experiments/corpus-audit/code-expansion.json")["working_demand"]
    natural_code = next(item for item in mixture if item["id"] == "natural_code")
    checked_code = next(item for item in mixture if item["id"] == "checked_code")
    if (
        code_expansion["base_horizon_tokens"] != target
        or code_expansion["natural_code_exposure_tokens"] != natural_code["exposure_tokens"]
        or code_expansion["natural_code_unique_preparation_tokens"]
        != natural_code["eligible_unique_token_preparation_target"]
        or code_expansion["checked_code_exposure_tokens"] != checked_code["exposure_tokens"]
        or code_expansion["checked_code_unique_preparation_tokens"]
        != checked_code["eligible_unique_token_preparation_target"]
    ):
        raise ValueError("code-expansion demand drifts from the working mixture")

    # Packets own run counts. Reconcile their components, not just the grand
    # total: two conflicting designs can both sum to the same reservation.
    pre = _load("experiments/main-data/data-study-packet.json")
    mid = _load("experiments/main-data/mid-training-study-packet.json")
    post = _load("experiments/main-data/post-training-study-packet.json")
    packet_budgets = {
        "pretraining": {
            "screening_training": pre["screening"]["training_gpu_hours"],
            "decay_training": pre["decay_study"]["training_gpu_hours"],
            "confirmation_training": pre["confirmation"]["training_gpu_hours"],
            "preparation_evaluation_recovery": pre["support_and_reserve"]["gpu_hours"],
        },
        "mid_training": {
            "proxy_screening_training": mid["screening"]["training_gpu_hours"],
            "objective_and_packing_training": mid["objective_and_packing_study"][
                "training_gpu_hours"
            ],
            "context_training": mid["context_study"]["training_gpu_hours"],
            "confirmation_training": mid["confirmation"]["training_gpu_hours"],
            "preparation_evaluation_recovery": mid["support_and_reserve"]["gpu_hours"],
        },
        "post_training": {
            "sft_paired_training": post["sft"]["training_gpu_hours"],
            "conditional_rl_prompt_feasibility": post["rl_feasibility"]["gpu_hours"],
            "final_self_sft_pilot": post["final_self_sft_pilot"]["training_gpu_hours"],
            "preparation_evaluation_recovery": post["support_and_reserve"]["gpu_hours"],
        },
    }
    for stage, components in packet_budgets.items():
        if components != compute["proposed_research_breakdown_gpu_hours"][stage]:
            raise ValueError(f"{stage} research components drift from the study packet")
        if sum(components.values()) != breakdown[stage]:
            raise ValueError(f"{stage} study packet does not match its reservation")

    study = plan["pretraining_data_study"]
    mirrors = (
        (study["proposed_screening"]["max_arms"], pre["screening"]["maximum_arms"]),
        (
            study["proposed_screening"]["per_arm_training_gpu_hour_cap"],
            pre["screening"]["per_arm_training_cap_gpu_hours"],
        ),
        (
            study["proposed_confirmation"]["per_arm_per_seed_training_gpu_hour_cap"],
            pre["confirmation"]["per_arm_per_seed_cap_gpu_hours"],
        ),
        (
            plan["context_extension"]["data_study"]["per_arm_training_gpu_hour_cap"],
            mid["context_study"]["per_arm_training_gpu_hour_cap"],
        ),
        (
            plan["post_training"]["data_study"]["sft_per_arm_per_seed_training_gpu_hour_cap"],
            post["sft"]["per_arm_per_seed_training_gpu_hour_cap"],
        ),
    )
    if any(summary != packet for summary, packet in mirrors):
        raise ValueError("plan run caps drift from study packets")
    if (
        mid["production_sequence"]["production_gpu_hours"]
        != reservations["capability_and_agentic_mid_training"]
    ):
        raise ValueError("mid-training production packet does not match its reservation")
    if (
        sum(compute["proposed_post_training_breakdown_gpu_hours"].values())
        != reservations["post_training"]
    ):
        raise ValueError("post-training production breakdown does not match its reservation")

    receipts = plan["input_receipts"]
    checked = 0
    for relative, expected in receipts.items():
        path = ROOT / relative
        # A missing receipt is a failure, not permission to validate fewer inputs.
        if not path.is_file():
            raise ValueError(f"missing input receipt: {relative}")
        if _sha256(path) != expected:
            raise ValueError(f"input receipt checksum mismatch: {relative}")
        checked += 1

    return {
        "requested_gpu_hours": compute["requested_total_gpu_hours"],
        "reservation_gpu_hours": sum(reservations.values()),
        "base_working_tokens": target,
        "base_unique_preparation_tokens": unique_target,
        "mid_research_gpu_hours": breakdown["mid_training"],
        "mid_production_gpu_hours": mid["production_sequence"]["production_gpu_hours"],
        "post_research_gpu_hours": breakdown["post_training"],
        "input_receipts_checked": checked,
        "training_authority": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", nargs="?", default=ROOT / "experiments/main-data/plan.json")
    args = parser.parse_args()
    print(json.dumps(validate(args.plan), indent=2, sort_keys=True))
