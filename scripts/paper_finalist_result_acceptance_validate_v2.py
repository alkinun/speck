"""Validate accepted finalist results, transitions, and retained checkpoint provenance."""

import argparse
from datetime import datetime
from pathlib import Path

from scripts.paper_finalist_continue import ordered_runs
from scripts.paper_finalist_result_acceptance_validate import (
    ROOT,
    file_sha256,
    load_object,
    resolve_path,
)
from scripts.paper_finalist_result_acceptance_validate import (
    validate_program as validate_acceptance_v1,
)
from speck.paper_finalist_analysis import collect_run_result

PROGRAM = ROOT / "research/paper-1/experiment_program.json"
PLAN = ROOT / "research/paper-1/finalist_analysis_v2.json"
MATERIALIZATION_CONTRACT = ROOT / "research/paper-1/finalist_materialization_v1.json"


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, default=PROGRAM)
    return parser.parse_args(argv)


def _stable_result(report):
    return {key: value for key, value in report.items() if key != "created_at"}


def _qualification(program, artifact_root):
    reference = program.get("finalist_qualification")
    if not isinstance(reference, dict):
        raise ValueError("finalist provenance requires the qualification reference")
    path = resolve_path(artifact_root, reference.get("result", ""))
    if not path.is_file() or file_sha256(path) != reference.get("sha256"):
        raise ValueError("finalist provenance qualification does not match its pin")
    qualification = load_object(path)
    if qualification.get("status") != reference.get("status"):
        raise ValueError("finalist provenance qualification has the wrong status")
    runs = qualification.get("runs")
    if not isinstance(runs, list):
        raise ValueError("finalist provenance qualification has no run inventory")
    return {entry.get("run"): entry for entry in runs}


def _validate_created_at(report):
    try:
        created_at = datetime.fromisoformat(report.get("created_at", ""))
    except (TypeError, ValueError) as error:
        raise ValueError("finalist result has an invalid collection timestamp") from error
    if created_at.tzinfo is None:
        raise ValueError("finalist result collection timestamp is not timezone-aware")


def validate_result_provenance(
    report,
    expected_name,
    qualification_runs,
    repository_root=ROOT,
    plan_path=PLAN,
    materialization_contract_path=MATERIALIZATION_CONTRACT,
):
    """Replay collection and require exact equality for every stable result field."""

    repository_root = Path(repository_root).resolve()
    expected = qualification_runs.get(expected_name)
    if not isinstance(expected, dict):
        raise ValueError(f"finalist provenance run is absent from qualification: {expected_name}")
    experiment = resolve_path(repository_root, expected.get("experiment", "")).resolve()
    checkpoint_directory = Path(expected.get("checkpoint_directory", "")).expanduser().resolve()
    if report.get("experiment") != str(experiment) or report.get("checkpoint", {}).get(
        "directory"
    ) != str(checkpoint_directory):
        raise ValueError(f"finalist result path provenance is invalid: {expected_name}")
    _validate_created_at(report)
    replay = collect_run_result(
        plan_path,
        materialization_contract_path,
        experiment,
        checkpoint_directory,
    )
    if _stable_result(report) != _stable_result(replay):
        raise ValueError(f"finalist result does not replay from retained evidence: {expected_name}")


def validate_program(
    program_path=PROGRAM,
    repository_root=ROOT,
    artifact_root=None,
    plan_path=None,
    materialization_contract_path=None,
):
    """Compose v1 ledger checks with deterministic checkpoint-to-result replay."""

    repository_root = Path(repository_root).resolve()
    artifact_root = Path(artifact_root).resolve() if artifact_root else repository_root
    plan_path = Path(plan_path) if plan_path else repository_root / PLAN.relative_to(ROOT)
    materialization_contract_path = (
        Path(materialization_contract_path)
        if materialization_contract_path
        else repository_root / MATERIALIZATION_CONTRACT.relative_to(ROOT)
    )
    base = validate_acceptance_v1(program_path, repository_root, artifact_root)
    program = load_object(program_path)
    qualification_runs = _qualification(program, artifact_root)
    evidence = program["finalist_evidence"]
    references = evidence["control_results"] + evidence["candidate_results"]
    names = ordered_runs()[: len(references)]
    for reference, expected_name in zip(references, names):
        result_path = resolve_path(artifact_root, reference["path"])
        validate_result_provenance(
            load_object(result_path),
            expected_name,
            qualification_runs,
            repository_root,
            plan_path,
            materialization_contract_path,
        )
    return {
        **base,
        "retained_checkpoint_provenance": len(references),
        "stable_result_replay": True,
    }


def main(argv=None):
    report = validate_program(arguments(argv).program)
    print(
        "Finalist accepted-result ledger and provenance: "
        f"{report['status']} ({report['accepted_results']} results, "
        f"{report['retained_checkpoint_provenance']} checkpoint replays, "
        f"next={report['next_run']})"
    )


if __name__ == "__main__":
    main()
