"""Validate the capability mid-training design without authorizing a run.

Run from the repository root::

    python experiments/main-data/check_mid_training_study.py \\
      experiments/main-data/mid-training-study-packet.json

The check verifies local evidence identities and budget arithmetic. It does not acquire data,
execute corpus or tool content, select a parent checkpoint or launch training.
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
    """Check design metadata only; this does not validate an executable trajectory."""
    packet = json.loads(Path(packet_path).resolve().read_text())
    if packet.get("format") != "speck_capability_mid_training_study_packet":
        raise ValueError("unsupported mid-training packet format")
    if packet.get("format_version") != 2:
        raise ValueError("unsupported mid-training packet version")
    if packet.get("status") != "design_only_not_training_authority":
        raise ValueError("mid-training packet must remain design-only")

    for entry in packet["source_of_truth"].values():
        if "sha256" in entry:
            _artifact(entry["path"], entry["sha256"])

    # These are separate comparisons, not a factorial sweep. Confirmation and
    # support remain reserved even if fewer screening candidates qualify.
    screening = packet["screening"]
    objective = packet["objective_and_packing_study"]
    context = packet["context_study"]
    confirmation = packet["confirmation"]
    support = packet["support_and_reserve"]
    if screening["maximum_arms"] != len(packet["arms"]):
        raise ValueError("screening arm count does not match arm definitions")
    if screening["training_gpu_hours"] != (
        screening["maximum_arms"] * screening["per_arm_training_gpu_hour_cap"]
    ):
        raise ValueError("screening budget does not match its arm cap")
    if confirmation["training_gpu_hours"] != (
        confirmation["arms"]
        * confirmation["new_paired_seed_count"]
        * confirmation["per_arm_per_seed_cap_gpu_hours"]
    ):
        raise ValueError("confirmation budget does not match its seed and arm caps")
    if objective["training_gpu_hours"] != (
        len(objective["arms"])
        * objective["paired_seed_count"]
        * objective["per_arm_training_gpu_hour_cap"]
    ):
        raise ValueError("objective/packing budget does not match its arm and seed caps")
    if context["training_gpu_hours"] != (
        len(context["arms"])
        * context["paired_seed_count"]
        * context["per_arm_training_gpu_hour_cap"]
    ):
        raise ValueError("context budget does not match its arm and seed caps")
    # Production percentages partition its own cap; research is not extra
    # exposure for the released model and must not be counted in these bands.
    production = packet["production_sequence"]
    if sum(production["planning_bands_percent"].values()) != 100:
        raise ValueError("mid-training production planning bands must total 100 percent")
    total = (
        screening["training_gpu_hours"]
        + objective["training_gpu_hours"]
        + context["training_gpu_hours"]
        + confirmation["training_gpu_hours"]
        + support["gpu_hours"]
    )
    if total != 300 or support["budget_check"] != (
        "120 proxy screening + 40 objective/packing + 40 context + 80 confirmation + 20 support = 300 mid-training-research GPU-hours."
    ):
        raise ValueError("mid-training research reservation must total 300 GPU-hours")
    if (
        packet["linked_downstream_evaluation"]["budget_owner"]
        != "post_training_research_150_gpu_hours"
    ):
        raise ValueError("downstream evaluation must remain outside the mid-training cap")
    if len(packet["trajectory_contract"]["required_fields"]) < 10:
        raise ValueError("trajectory contract must bind environment and outcome fields")
    if not packet["boundary"].startswith("No arm is launch-ready"):
        raise ValueError("packet boundary must keep launch authority outside the design packet")
    return {
        "format": packet["format"],
        "status": packet["status"],
        "candidate_options": len(packet["arms"]),
        "screening_gpu_hours": screening["training_gpu_hours"],
        "objective_gpu_hours": objective["training_gpu_hours"],
        "confirmation_gpu_hours": confirmation["training_gpu_hours"],
        "context_gpu_hours": context["training_gpu_hours"],
        "support_gpu_hours": support["gpu_hours"],
        "total_gpu_hours": total,
        "training_admitted": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.packet), indent=2, sort_keys=True))
