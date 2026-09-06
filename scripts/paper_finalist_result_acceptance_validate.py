"""Validate every automatically accepted finalist result and transition."""

import argparse
import hashlib
import json
from pathlib import Path

from scripts.paper_finalist_continue import candidate_runs, control_runs, ordered_runs
from scripts.paper_finalist_source_coverage_validate import validate_result as validate_sources

ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "research/paper-1/experiment_program.json"
AUTOMATION = ROOT / "research/paper-1/finalist_automation_v2.json"
TRANSITIONS = Path("results/Speck-Paper1/finalist-transitions")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, default=PROGRAM)
    return parser.parse_args(argv)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def resolve_path(artifact_root, value):
    path = Path(value)
    return path if path.is_absolute() else artifact_root / path


def validate_reference(reference, artifact_root, expected_status):
    if not isinstance(reference, dict):
        raise ValueError("finalist acceptance reference is absent")
    path = resolve_path(artifact_root, reference.get("path", ""))
    if (
        not path.is_file()
        or file_sha256(path) != reference.get("sha256")
        or reference.get("status") != expected_status
    ):
        raise ValueError(f"finalist acceptance reference is invalid: {path}")
    value = load_object(path)
    if value.get("status") != expected_status:
        raise ValueError(f"finalist acceptance artifact has the wrong status: {path}")
    return path, value


def validate_program(program_path, repository_root=ROOT, artifact_root=None):
    repository_root = Path(repository_root).resolve()
    artifact_root = Path(artifact_root).resolve() if artifact_root else repository_root
    program = load_object(program_path)
    evidence = program.get("finalist_evidence", {})
    controls = evidence.get("control_results")
    candidates = evidence.get("candidate_results")
    if not isinstance(controls, list) or not isinstance(candidates, list):
        raise ValueError("finalist acceptance program has invalid result lists")
    if not 0 <= len(controls) <= 6 or not 0 <= len(candidates) <= 6:
        raise ValueError("finalist acceptance result counts exceed the frozen design")
    if len(controls) < 6 and candidates:
        raise ValueError("finalist acceptance violates control-first ordering")

    expected_names = control_runs()[: len(controls)] + candidate_runs()[: len(candidates)]
    references = controls + candidates
    automation_hash = file_sha256(repository_root / AUTOMATION.relative_to(ROOT))
    accepted = []
    for index, (reference, expected_name) in enumerate(zip(references, expected_names)):
        result_path = resolve_path(artifact_root, reference.get("path", ""))
        if (
            not result_path.is_file()
            or file_sha256(result_path) != reference.get("sha256")
            or reference.get("status") != "complete_qualified"
        ):
            raise ValueError(f"accepted finalist result reference is invalid: {expected_name}")
        report = load_object(result_path)
        coverage = validate_sources(report)
        expected_pair = index if index < 6 else index - 6
        expected_arm = "dense_global_param_match" if index < 6 else "five_cache_kda_gqa"
        if (
            report.get("run") != expected_name
            or report.get("arm_id") != expected_arm
            or report.get("pair", {}).get("pair") != expected_pair
            or reference.get("pair") != expected_pair
        ):
            raise ValueError(f"accepted finalist result identity is invalid: {expected_name}")
        transition_path = artifact_root / TRANSITIONS / f"{expected_name}.json"
        transition = load_object(transition_path)
        expected_next = ordered_runs()[index + 1] if index + 1 < len(ordered_runs()) else None
        expected_result_path = result_path.relative_to(artifact_root).as_posix()
        if (
            transition.get("format") != "speck_paper_finalist_automatic_transition"
            or transition.get("format_version") != 2
            or transition.get("status") != "complete"
            or transition.get("completed_run") != expected_name
            or transition.get("result")
            != {"path": expected_result_path, "sha256": reference["sha256"]}
            or transition.get("next_run") != expected_next
            or transition.get("quality_dependent_branching") is not False
            or transition.get("trigger_disabled_before_collection") is not True
            or transition.get("polling") is not False
            or transition.get("automatic_retry") is not False
            or transition.get("automation_contract_sha256") != automation_hash
        ):
            raise ValueError(f"accepted finalist transition is invalid: {expected_name}")
        accepted.append(
            {
                "run": expected_name,
                "pair": expected_pair,
                "arm": expected_arm,
                "result_sha256": reference["sha256"],
                "sources": coverage["sources"],
                "next_run": expected_next,
            }
        )

    target = evidence.get("time_to_quality_target")
    analysis = evidence.get("analysis_result")
    if len(controls) < 6:
        if target is not None or analysis is not None:
            raise ValueError("finalist acceptance found target or analysis before six controls")
    else:
        validate_reference(
            target,
            artifact_root,
            "locked_from_six_controls_before_candidates",
        )
    if len(candidates) < 6:
        if analysis is not None:
            raise ValueError("finalist acceptance found analysis before six candidates")
    else:
        validate_reference(
            analysis,
            artifact_root,
            "complete_crossed_factor_finalist_language_evidence_no_standalone_promotion",
        )

    count = len(references)
    expected_next = ordered_runs()[count] if count < 12 else None
    expected_status = (
        "qualified_unexecuted" if count == 0 else "complete" if count == 12 else "in_progress"
    )
    if evidence.get("next_run") != expected_next or evidence.get("status") != expected_status:
        raise ValueError("finalist acceptance program status or next run is invalid")
    if accepted and accepted[-1]["next_run"] != expected_next:
        raise ValueError("latest finalist transition does not match the program next run")
    transitions_dir = artifact_root / TRANSITIONS
    observed_transitions = (
        {path.stem for path in transitions_dir.glob("*.json")}
        if transitions_dir.is_dir()
        else set()
    )
    if observed_transitions != set(expected_names):
        raise ValueError("finalist acceptance transition inventory does not match accepted results")
    failed = evidence.get("failed_attempts", ())
    rerun = evidence.get("active_rerun", {})
    if (
        len(failed) != 1
        or failed[0].get("status") != "operator_interrupted_after_step_1_no_checkpoint_or_result"
        or rerun.get("attempt") != 2
        or rerun.get("status") != "frozen_after_operator_interruption_before_identical_restart"
    ):
        raise ValueError("finalist acceptance lost the failed-attempt or rerun history")
    return {
        "status": "valid",
        "accepted_results": len(accepted),
        "controls": len(controls),
        "candidates": len(candidates),
        "next_run": expected_next,
        "target_locked": target is not None,
        "analysis_complete": analysis is not None,
        "complete_source_coverage": all(entry["sources"] == 11 for entry in accepted),
    }


def main(argv=None):
    report = validate_program(arguments(argv).program)
    print(
        "Finalist accepted-result ledger: "
        f"{report['status']} ({report['accepted_results']} results, next={report['next_run']})"
    )


if __name__ == "__main__":
    main()
