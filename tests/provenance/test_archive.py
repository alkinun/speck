import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from speck.provenance.archive import locate_original, verify_archive
from speck.provenance.repository import repository_root


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def fixture_archive(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Archive fixture")
    git(tmp_path, "config", "user.email", "fixture@example.invalid")
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n")
    result = tmp_path / "result.json"
    result.write_text('{"measured": 1}\n')
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "Original result")
    revision = git(tmp_path, "rev-parse", "HEAD")
    archived = tmp_path / "archive/pregrant-history/result.json"
    archived.parent.mkdir(parents=True)
    archived.write_bytes(result.read_bytes())
    files = []
    for path in (result, source):
        entry = {
            "original_path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        if path == result:
            entry["archived_path"] = "archive/pregrant-history/result.json"
        files.append(entry)
    manifest = {
        "format": "speck_research_archive",
        "format_version": 1,
        "revision": revision,
        "files": files,
    }
    (tmp_path / "archive/manifest.json").write_text(json.dumps(manifest))
    return source, archived


def test_archive_preserves_original_evidence_while_source_evolves(tmp_path):
    source, archived = fixture_archive(tmp_path)
    source.write_text("VALUE = 2\n")
    assert verify_archive(tmp_path)["inventoried_files"] == 2
    archived.write_text('{"measured": 2}\n')
    with pytest.raises(ValueError, match="archived bytes changed"):
        verify_archive(tmp_path)


def test_archive_rejects_rehashed_evidence_and_uninventoried_files(tmp_path):
    _, archived = fixture_archive(tmp_path)
    path = tmp_path / "archive/manifest.json"
    manifest = json.loads(path.read_text())
    archived.write_text("replacement")
    manifest["files"][0]["sha256"] = hashlib.sha256(archived.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="original Git identity mismatch"):
        verify_archive(tmp_path)
    archived.with_name("unrecorded.json").write_text("{}")
    with pytest.raises(ValueError, match="physical archive files"):
        verify_archive(tmp_path)


def test_archived_markdown_references_have_original_or_revision_locations():
    root = repository_root()
    manifest = json.loads((root / "archive/manifest.json").read_text())
    paths = {entry["original_path"] for entry in manifest["files"]}
    directories = {str(parent) for path in paths for parent in Path(path).parents}
    resolved = {}
    for entry in manifest["files"]:
        if not entry["original_path"].endswith(".md"):
            continue
        document = root / entry["archived_path"]
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", document.read_text()):
            target = target.split("#", 1)[0]
            if not target or re.match(r"^[a-z]+:", target):
                continue
            original = os.path.normpath(str(Path(entry["original_path"]).parent / target))
            if original not in paths and original not in directories:
                if original not in resolved:
                    resolved[original] = locate_original(original, root)
                assert resolved[original]["revision"]


def test_current_execution_successor_preserves_phase_budgets_and_dependencies():
    root = repository_root()
    previous = json.loads(
        (root / "archive/pregrant-history/research/flagship/plan_v2.json").read_text()
    )
    current = json.loads((root / "research/flagship/plan_v3.json").read_text())
    assert current["budget"] == previous["budget"]
    assert [(p["id"], p["gpu_hours"], p.get("depends_on")) for p in current["phases"]] == [
        (p["id"], p["gpu_hours"], p.get("depends_on")) for p in previous["phases"]
    ]
    p6 = next(p for p in current["phases"] if p["id"] == "P6")
    assert sum(p6["initial_sub_budgets_gpu_hours"].values()) == p6["gpu_hours"]
    assert all((root / path).is_file() for path in current["active_contracts"].values())
