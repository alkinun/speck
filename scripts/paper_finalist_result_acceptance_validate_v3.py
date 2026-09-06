"""Validate finalist evidence through checkpoint replay and append-only Git commits."""

import argparse
import json
import subprocess
from pathlib import Path

from scripts.paper_finalist_continue import ordered_runs
from scripts.paper_finalist_result_acceptance_validate import ROOT, load_object, resolve_path
from scripts.paper_finalist_result_acceptance_validate_v2 import (
    PROGRAM,
)
from scripts.paper_finalist_result_acceptance_validate_v2 import (
    validate_program as validate_acceptance_v2,
)

TRANSITIONS = Path("results/Speck-Paper1/finalist-transitions")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, default=PROGRAM)
    return parser.parse_args(argv)


def _git(root, *arguments, check=True):
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=check,
        capture_output=True,
        text=True,
    )


def _relative(root, path):
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"finalist Git evidence is outside the repository: {path}") from error


def _commits_for_path(root, path):
    output = _git(root, "log", "--format=%H", "--", path).stdout
    return [line for line in output.splitlines() if line]


def _require_write_once_path(root, path, commit, name):
    history = _commits_for_path(root, path)
    if history != [commit]:
        raise ValueError(f"finalist {name} must be write-once in Git: {path}")
    if _git(root, "diff", "--quiet", commit, "--", path, check=False).returncode != 0:
        raise ValueError(f"finalist {name} differs from its committed bytes: {path}")


def _commit_json(root, commit, path):
    output = _git(root, "show", f"{commit}:{path}").stdout
    value = json.loads(output)
    if not isinstance(value, dict):
        raise ValueError(f"finalist Git object is not a JSON object: {path}")
    return value


def _expected_subject(index):
    if index < 5:
        return f"Register Paper 1 finalist control {index}"
    if index == 5:
        return "Lock Paper 1 finalist control target"
    if index < 11:
        return f"Register Paper 1 finalist candidate {index - 6}"
    return "Complete Paper 1 finalist paired analysis"


def _validate_program_snapshot(snapshot, evidence, index):
    observed = snapshot.get("finalist_evidence", {})
    controls = min(index + 1, 6)
    candidates = max(index - 5, 0)
    expected_next = ordered_runs()[index + 1] if index < 11 else None
    expected_status = "complete" if index == 11 else "in_progress"
    if (
        observed.get("control_results") != evidence["control_results"][:controls]
        or observed.get("candidate_results") != evidence["candidate_results"][:candidates]
        or observed.get("time_to_quality_target")
        != (evidence["time_to_quality_target"] if index >= 5 else None)
        or observed.get("analysis_result") != (evidence["analysis_result"] if index == 11 else None)
        or observed.get("next_run") != expected_next
        or observed.get("status") != expected_status
    ):
        raise ValueError(f"finalist Git program snapshot is invalid at transition {index}")


def validate_git_provenance(program_path, artifact_root=ROOT, git_root=None):
    """Require one exact append-only commit for every accepted transition."""

    artifact_root = Path(artifact_root).resolve()
    git_root = Path(git_root).resolve() if git_root else artifact_root
    program_path = Path(program_path).resolve()
    program_relative = _relative(git_root, program_path)
    program = load_object(program_path)
    evidence = program["finalist_evidence"]
    references = evidence["control_results"] + evidence["candidate_results"]
    names = ordered_runs()[: len(references)]
    transition_commits = []
    for index, (reference, name) in enumerate(zip(references, names)):
        result_relative = _relative(git_root, resolve_path(artifact_root, reference["path"]))
        transition_relative = _relative(git_root, artifact_root / TRANSITIONS / f"{name}.json")
        result_commits = _commits_for_path(git_root, result_relative)
        transition_history = _commits_for_path(git_root, transition_relative)
        if len(result_commits) != 1 or len(transition_history) != 1:
            raise ValueError(f"finalist result and transition must be write-once in Git: {name}")
        commit = transition_history[0]
        if result_commits[0] != commit:
            raise ValueError(f"finalist result and transition were not committed together: {name}")
        _require_write_once_path(git_root, result_relative, commit, "result")
        _require_write_once_path(git_root, transition_relative, commit, "transition")
        subject = _git(git_root, "show", "-s", "--format=%s", commit).stdout.strip()
        if subject != _expected_subject(index):
            raise ValueError(f"finalist transition commit has the wrong subject: {name}")
        changed = {
            line
            for line in _git(
                git_root,
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "-r",
                commit,
            ).stdout.splitlines()
            if line
        }
        expected_changed = {program_relative, result_relative, transition_relative}
        if index == 5:
            target_relative = _relative(
                git_root,
                resolve_path(artifact_root, evidence["time_to_quality_target"]["path"]),
            )
            expected_changed.add(target_relative)
            _require_write_once_path(git_root, target_relative, commit, "target lock")
        if index == 11:
            analysis_relative = _relative(
                git_root,
                resolve_path(artifact_root, evidence["analysis_result"]["path"]),
            )
            expected_changed.add(analysis_relative)
            _require_write_once_path(git_root, analysis_relative, commit, "analysis")
        if changed != expected_changed:
            raise ValueError(f"finalist transition commit has an invalid file set: {name}")
        _validate_program_snapshot(
            _commit_json(git_root, commit, program_relative),
            evidence,
            index,
        )
        if (
            transition_commits
            and _git(
                git_root,
                "merge-base",
                "--is-ancestor",
                transition_commits[-1],
                commit,
                check=False,
            ).returncode
            != 0
        ):
            raise ValueError("finalist transition commits are not in frozen ancestry order")
        transition_commits.append(commit)
    return transition_commits


def validate_program(
    program_path=PROGRAM,
    repository_root=ROOT,
    artifact_root=None,
    git_root=None,
    plan_path=None,
    materialization_contract_path=None,
):
    """Compose v2 evidence replay with exact append-only event-commit checks."""

    artifact_root = (
        Path(artifact_root).resolve() if artifact_root else Path(repository_root).resolve()
    )
    base = validate_acceptance_v2(
        program_path,
        repository_root,
        artifact_root,
        plan_path,
        materialization_contract_path,
    )
    commits = validate_git_provenance(program_path, artifact_root, git_root)
    return {
        **base,
        "git_transition_commits": len(commits),
        "append_only_commit_provenance": True,
    }


def main(argv=None):
    report = validate_program(arguments(argv).program)
    print(
        "Finalist accepted-result ledger, checkpoint, and Git provenance: "
        f"{report['status']} ({report['accepted_results']} results, "
        f"{report['retained_checkpoint_provenance']} checkpoint replays, "
        f"{report['git_transition_commits']} event commits, next={report['next_run']})"
    )


if __name__ == "__main__":
    main()
