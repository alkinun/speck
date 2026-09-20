"""Validate the post-training study design without authorizing a run.

Run from the repository root::

    python experiments/main-data/check_post_training_study.py \\
      experiments/main-data/post-training-study-packet.json

The check verifies local evidence identities and budget arithmetic. It does not acquire data,
execute tasks or tools, update a policy, select examples or launch training.
"""

import argparse
import json
from pathlib import Path

from speck.provenance.io import file_sha256

ROOT = Path(__file__).resolve().parents[2]


def _artifact(path, expected):
    resolved = (ROOT / path).resolve() if not Path(path).is_absolute() else Path(path)
    if file_sha256(resolved) != expected:
        raise ValueError(f"artifact checksum mismatch: {resolved}")
    return resolved


def validate(packet_path):
    """Check study ceilings without admitting data, rewards or policy updates."""
    packet = json.loads(Path(packet_path).resolve().read_text())
    if packet.get("format") != "speck_post_training_data_study_packet":
        raise ValueError("unsupported post-training packet format")
    if packet.get("format_version") != 2:
        raise ValueError("unsupported post-training packet version")
    if packet.get("status") != "design_only_not_training_authority":
        raise ValueError("post-training packet must remain design-only")

    for entry in packet["source_of_truth"].values():
        if "sha256" in entry:
            _artifact(entry["path"], entry["sha256"])

    sft = packet["sft"]
    if len(sft["arms"]) != 2:
        raise ValueError("SFT comparison must contain exactly two arms")
    expected_sft = (
        len(sft["arms"]) * sft["paired_seed_count"] * sft["per_arm_per_seed_training_gpu_hour_cap"]
    )
    if sft["training_gpu_hours"] != expected_sft:
        raise ValueError("SFT budget does not match arm and seed caps")
    # This is a research comparison. Production self-SFT must fit the separate
    # post-training ceiling rather than inheriting an additional allocation.
    self_sft = packet["final_self_sft_pilot"]
    if self_sft["training_gpu_hours"] != (
        len(self_sft["arms"])
        * self_sft["paired_seed_count"]
        * self_sft["per_arm_training_gpu_hour_cap"]
    ):
        raise ValueError("self-SFT pilot budget does not match arm and seed caps")
    support = packet["support_and_reserve"]
    rl = packet["rl_feasibility"]
    total = (
        sft["training_gpu_hours"]
        + rl["gpu_hours"]
        + self_sft["training_gpu_hours"]
        + support["gpu_hours"]
    )
    if total != 150 or support["budget_check"] != (
        "80 SFT comparison + 20 RL feasibility + 30 self-SFT pilot + 20 support = 150 post-training-research GPU-hours."
    ):
        raise ValueError("post-training research reservation must total 150 GPU-hours")
    # Prompt/verifier feasibility cannot establish RL learning or adaptation:
    # this slot samples a fixed policy and explicitly excludes gradient updates.
    if rl["status"] != "conditional_fixed_policy_only":
        raise ValueError("RL slot must remain fixed-policy feasibility only")
    if "policy updates" not in rl["boundary"]:
        raise ValueError("RL boundary must exclude policy updates")
    if not packet["boundary"].startswith("No post-training arm is launch-ready"):
        raise ValueError("packet boundary must keep launch authority outside the design packet")
    return {
        "format": packet["format"],
        "status": packet["status"],
        "sft_arms": len(sft["arms"]),
        "sft_gpu_hours": sft["training_gpu_hours"],
        "rl_feasibility_gpu_hours": rl["gpu_hours"],
        "self_sft_pilot_gpu_hours": self_sft["training_gpu_hours"],
        "support_gpu_hours": support["gpu_hours"],
        "total_gpu_hours": total,
        "training_admitted": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.packet), indent=2, sort_keys=True))
