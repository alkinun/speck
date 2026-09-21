"""Validate the bounded architecture study design without authorizing a run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from speck.provenance.io import file_sha256

ROOT = Path(__file__).resolve().parents[2]


def _artifact(path: str, expected: str) -> None:
    resolved = (ROOT / path).resolve() if not Path(path).is_absolute() else Path(path)
    if file_sha256(resolved) != expected:
        raise ValueError(f"artifact checksum mismatch: {resolved}")


def validate(packet_path: str | Path) -> dict[str, object]:
    packet = json.loads(Path(packet_path).resolve().read_text())
    if packet.get("format") != "speck_architecture_efficiency_study_packet":
        raise ValueError("unsupported architecture study packet format")
    if packet.get("format_version") != 1:
        raise ValueError("unsupported architecture study packet version")
    if packet.get("status") != "deferred_to_a_later_allocation_design_preserved":
        raise ValueError("architecture study packet must remain deferred")

    for entry in packet["source_of_truth"].values():
        _artifact(entry["path"], entry["sha256"])

    # The comparison is deferred, so the packet must draw nothing on this
    # allocation. The released hours stay recorded here so the deferral is
    # auditable rather than a silent disappearance from the reservation table.
    budget = packet["budget"]
    if budget["total_gpu_hours"] != 0 or any(
        budget[key] != 0
        for key in (
            "paired_training_gpu_hours",
            "profiling_gpu_hours",
            "qualification_evaluation_recovery_gpu_hours",
        )
    ):
        raise ValueError("deferred architecture study must draw zero GPU-hours")
    if budget["released_gpu_hours"] != 200:
        raise ValueError("architecture study must record the 200 released GPU-hours")
    deferral = packet["deferral"]
    if not deferral.get("claim_consequence", "").startswith("No architecture superiority"):
        raise ValueError("deferral must forbid an architecture claim in the report")

    arms = packet["arms"]
    if len(arms) != 2 or {arm["role"] for arm in arms} != {"reference", "single_control"}:
        raise ValueError("architecture study must contain one reference and one single control")
    training = packet["training"]
    if training["paired_seed_count"] != 1 or training["per_arm_gpu_hour_cap"] != 60:
        raise ValueError("architecture training caps must remain one seed and 60 hours per arm")
    profiling = packet["profiling"]
    if profiling["lengths"] != [512, 2048, 3840] or profiling["batch_sizes"] != [1, 8]:
        raise ValueError("profiling lengths or batch sizes changed from the bounded design")
    if not packet["boundary"].startswith("No architecture result is launch-ready"):
        raise ValueError(
            "architecture packet boundary must keep launch authority outside the design"
        )
    return {
        "format": packet["format"],
        "status": packet["status"],
        "arms": len(arms),
        "total_gpu_hours": budget["total_gpu_hours"],
        "training_admitted": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.packet), indent=2, sort_keys=True))
