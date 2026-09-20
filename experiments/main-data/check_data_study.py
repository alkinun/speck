"""Validate the pretraining data-study design packet without authorizing a run.

Run from the repository root::

    python experiments/main-data/check_data_study.py experiments/main-data/data-study-packet.json

The check verifies local evidence identities and design arithmetic. It does not acquire sources,
execute corpus content, select eligible data or launch training.
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
    packet_path = Path(packet_path).resolve()
    packet = json.loads(packet_path.read_text())
    if packet.get("format") != "speck_pretraining_data_study_packet":
        raise ValueError("unsupported data-study packet format")
    if packet.get("format_version") != 1:
        raise ValueError("unsupported data-study packet version")
    if packet.get("status") != "design_only_not_training_authority":
        raise ValueError("data-study packet must remain design-only")

    sources = packet["source_of_truth"]
    for entry in sources.values():
        _artifact(entry["path"], entry["sha256"]) if "sha256" in entry else None

    screening = packet["screening"]
    confirmation = packet["confirmation"]
    support = packet["support_and_reserve"]
    if screening["maximum_arms"] != len(packet["arms"]):
        raise ValueError("screening arm count does not match arm definitions")
    if screening["training_gpu_hours"] != (
        screening["maximum_arms"] * screening["per_arm_training_cap_gpu_hours"]
    ):
        raise ValueError("screening budget does not match its arm cap")
    if confirmation["arms"] != 2:
        raise ValueError("confirmation must compare baseline and one candidate")
    if confirmation["training_gpu_hours"] != (
        confirmation["arms"]
        * confirmation["new_paired_seed_count"]
        * confirmation["per_arm_per_seed_cap_gpu_hours"]
    ):
        raise ValueError("confirmation budget does not match its seed and arm caps")
    total = (
        screening["training_gpu_hours"] + confirmation["training_gpu_hours"] + support["gpu_hours"]
    )
    if total != 600 or support["budget_check"] != (
        "120 screening + 360 confirmation + 120 support = 600 pretraining-data-research GPU-hours."
    ):
        raise ValueError("pretraining data-study reservation must total 600 GPU-hours")
    required = packet["required_gates_before_first_arm"]
    if not any("source-use" in gate and "human" in gate for gate in required):
        raise ValueError("source-use gate must require named human decisions")
    if not packet["boundary"].startswith("No arm is launch-ready"):
        raise ValueError("packet boundary must keep launch authority outside the design packet")
    return {
        "format": packet["format"],
        "status": packet["status"],
        "arms": len(packet["arms"]),
        "screening_gpu_hours": screening["training_gpu_hours"],
        "confirmation_gpu_hours": confirmation["training_gpu_hours"],
        "support_gpu_hours": support["gpu_hours"],
        "total_gpu_hours": total,
        "training_admitted": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.packet), indent=2, sort_keys=True))
