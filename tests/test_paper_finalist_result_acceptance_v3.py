import json
import subprocess
from pathlib import Path

import pytest

from scripts.paper_finalist_continue import ordered_runs
from scripts.paper_finalist_result_acceptance_validate_v2 import (
    validate_program as validate_acceptance_v2,
)
from scripts.paper_finalist_result_acceptance_validate_v3 import validate_program
from tests.test_paper_finalist_result_acceptance_v2 import (
    atomic_json,
    one_control_state,
    rehash_result,
)

repository_root = Path(__file__).parents[1]


def git(root, *arguments):
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


def prepared_repository(tmp_path):
    program_path, result_path, transition_path, checkpoint = one_control_state(tmp_path)
    canonical_program = tmp_path / "research" / "paper-1" / "experiment_program.json"
    final_program = json.loads(program_path.read_text(encoding="utf-8"))
    initial_program = json.loads(program_path.read_text(encoding="utf-8"))
    initial_evidence = initial_program["finalist_evidence"]
    initial_evidence["status"] = "qualified_unexecuted"
    initial_evidence["control_results"] = []
    initial_evidence["next_run"] = ordered_runs()[0]
    atomic_json(canonical_program, initial_program)
    program_path.unlink()
    (tmp_path / ".gitignore").write_text("checkpoint/\n", encoding="utf-8")
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "speck-fixture@example.invalid")
    git(tmp_path, "config", "user.name", "Speck fixture")
    git(
        tmp_path,
        "add",
        ".gitignore",
        "qualification.json",
        "research/paper-1/experiment_program.json",
    )
    git(tmp_path, "commit", "-q", "-m", "Initialize finalist fixture")
    atomic_json(canonical_program, final_program)
    return canonical_program, result_path, transition_path, checkpoint


def commit_event(root, program_path, result_path, transition_path, *, subject=None, extra=False):
    paths = [program_path, result_path, transition_path]
    if extra:
        extra_path = root / "unrelated.txt"
        extra_path.write_text("unrelated\n", encoding="utf-8")
        paths.append(extra_path)
    git(root, "add", *[path.relative_to(root).as_posix() for path in paths])
    git(
        root,
        "commit",
        "-q",
        "-m",
        subject or "Register Paper 1 finalist control 0",
    )


def test_v3_accepts_exact_append_only_event_commit(tmp_path):
    program, result, transition, _ = prepared_repository(tmp_path)
    commit_event(tmp_path, program, result, transition)
    report = validate_program(program, repository_root, tmp_path, tmp_path)
    assert report["git_transition_commits"] == 1
    assert report["append_only_commit_provenance"] is True


def test_v3_rejects_uncommitted_state_that_v2_accepts(tmp_path):
    program, _, _, _ = prepared_repository(tmp_path)
    assert validate_acceptance_v2(program, repository_root, tmp_path)["status"] == "valid"
    with pytest.raises(ValueError, match="write-once"):
        validate_program(program, repository_root, tmp_path, tmp_path)


def test_v3_rejects_unrelated_file_in_event_commit(tmp_path):
    program, result, transition, _ = prepared_repository(tmp_path)
    commit_event(tmp_path, program, result, transition, extra=True)
    assert validate_acceptance_v2(program, repository_root, tmp_path)["status"] == "valid"
    with pytest.raises(ValueError, match="file set"):
        validate_program(program, repository_root, tmp_path, tmp_path)


def test_v3_rejects_wrong_event_commit_subject(tmp_path):
    program, result, transition, _ = prepared_repository(tmp_path)
    commit_event(tmp_path, program, result, transition, subject="Ambiguous result update")
    assert validate_acceptance_v2(program, repository_root, tmp_path)["status"] == "valid"
    with pytest.raises(ValueError, match="wrong subject"):
        validate_program(program, repository_root, tmp_path, tmp_path)


def test_v3_rejects_later_result_rewrite_that_v2_accepts(tmp_path):
    program, result, transition, _ = prepared_repository(tmp_path)
    commit_event(tmp_path, program, result, transition)
    report = json.loads(result.read_text(encoding="utf-8"))
    report["created_at"] = "2026-09-07T00:00:00+00:00"
    atomic_json(result, report)
    rehash_result(program, result, transition)
    git(
        tmp_path,
        "add",
        program.relative_to(tmp_path).as_posix(),
        result.relative_to(tmp_path).as_posix(),
        transition.relative_to(tmp_path).as_posix(),
    )
    git(tmp_path, "commit", "-q", "-m", "Rewrite collected timestamp")
    assert validate_acceptance_v2(program, repository_root, tmp_path)["status"] == "valid"
    with pytest.raises(ValueError, match="write-once"):
        validate_program(program, repository_root, tmp_path, tmp_path)


def test_v3_rejects_uncommitted_transition_rewrite_that_v2_accepts(tmp_path):
    program, result, transition, _ = prepared_repository(tmp_path)
    commit_event(tmp_path, program, result, transition)
    value = json.loads(transition.read_text(encoding="utf-8"))
    value["created_at"] = "2026-09-07T00:00:00+00:00"
    atomic_json(transition, value)
    assert validate_acceptance_v2(program, repository_root, tmp_path)["status"] == "valid"
    with pytest.raises(ValueError, match="committed bytes"):
        validate_program(program, repository_root, tmp_path, tmp_path)
