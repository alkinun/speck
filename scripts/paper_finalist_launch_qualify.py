"""Record the live Paper 1 finalist launch gate without launching training."""

import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from scripts.paper_baseline_preflight import atomic_json, repository_revision
from scripts.paper_finalist_continue import (
    ROOT,
    _live_gate,
    _unit_stem,
    expected_next,
    file_sha256,
    load_object,
    ordered_runs,
    validate_automation_contract,
)
from speck.paper import validate_paper_program

PROGRAM = ROOT / "research/paper-1/experiment_program.json"
LAUNCH = ROOT / "research/paper-1/finalist_launch_v1.json"
AUTOMATION = ROOT / "research/paper-1/finalist_automation_v1.json"


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def validate_initial_state(program):
    evidence = program.get("finalist_evidence", {})
    first = ordered_runs()[0]
    if (
        evidence.get("status") != "qualified_unexecuted"
        or evidence.get("control_results") != []
        or evidence.get("candidate_results") != []
        or evidence.get("time_to_quality_target") is not None
        or evidence.get("analysis_result") is not None
        or evidence.get("next_run") != first
        or expected_next(evidence) != first
    ):
        raise ValueError("finalist program is not in its frozen empty initial state")
    return first


def finalist_units():
    names = []
    for run_name in ordered_runs():
        stem = _unit_stem(run_name)
        names.extend(
            [
                f"{stem}-launch.service",
                f"{stem}-launch.timer",
                f"{stem}-finalize.path",
                f"{stem}-finalize.service",
            ]
        )
    return names


def active_units():
    active = []
    for unit in finalist_units():
        result = subprocess.run(
            ["systemctl", "--user", "is-active", unit],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.stdout.strip() == "active":
            active.append(unit)
    return active


def prepare(output):
    revision = repository_revision()
    automation = validate_automation_contract()
    program = load_object(PROGRAM)
    first = validate_initial_state(program)
    validation = validate_paper_program(ROOT / "research/paper-1")
    if validation.get("status") != "valid_hypotheses_only":
        raise ValueError("paper program validation did not pass")
    conflicts = active_units()
    if conflicts:
        raise ValueError(f"finalist units already active: {conflicts}")
    live = _live_gate(first)
    report = {
        "format": "speck_paper_finalist_launch_qualification",
        "format_version": 1,
        "status": "qualified_initial_control_launch_authorized_no_promotion_authority",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "paper_id": automation["paper_id"],
        "inputs": {
            "launch_contract": {
                "path": LAUNCH.relative_to(ROOT).as_posix(),
                "sha256": file_sha256(LAUNCH),
            },
            "automation_contract": {
                "path": AUTOMATION.relative_to(ROOT).as_posix(),
                "sha256": file_sha256(AUTOMATION),
            },
            "program": {
                "path": PROGRAM.relative_to(ROOT).as_posix(),
                "sha256_before_qualification_registration": file_sha256(PROGRAM),
                "validation_status": validation["status"],
            },
        },
        "initial_state": program["finalist_evidence"],
        "initial_run": first,
        "active_finalist_units": conflicts,
        "live_gate": live,
        "services": {
            "helmet_download": "inactive",
            "finalist_units_active": 0,
        },
        "implementation": {
            "runner_revision": revision,
            "runner_sha256": file_sha256(__file__),
            "automation_runner_sha256": file_sha256(
                ROOT / automation["implementation"]["runner"]
            ),
        },
        "decision": {
            "initial_control_launch_authorized": True,
            "event_driven_successors_authorized": True,
            "training_authorized": True,
            "helmet_concurrency_authorized": False,
            "quality_dependent_branching_authorized": False,
            "automatic_retry_authorized": False,
            "component_attribution_authorized": False,
            "architecture_promotion_authorized": False,
            "novelty_claim_authorized": False,
            "release_claim_authorized": False,
            "paper_scale_authorized": False,
            "next_action": "commit and register this qualification, revalidate the clean repository, then launch only the exact initial dense control through the event runner",
        },
    }
    atomic_json(output, report)
    return report


def main(argv=None):
    args = arguments(argv)
    report = prepare(args.output)
    print(f"Paper 1 finalist launch: {report['status']}")


if __name__ == "__main__":
    main()
