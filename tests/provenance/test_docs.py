import json
import re

import pytest

from speck.provenance.repository import repository_root
from tests.reference import historical_repository

root = repository_root()


def test_local_markdown_links_exist():
    # Partial snapshots preserve original relative links, not a relocated full tree.
    snapshot_origins = {}
    for manifest_path in (root / "research/history").glob("*/manifest.json"):
        manifest = json.loads(manifest_path.read_text())
        snapshot_origins.update(
            {
                root / entry["preserved_path"]: root / entry["original_path"]
                for entry in manifest["files"]
            }
        )
    documents = (
        sorted(root.glob("*.md"))
        + sorted((root / "docs").rglob("*.md"))
        + [root / "experiments" / "README.md"]
        + sorted((root / "paper").rglob("*.md"))
        + sorted((root / "research").rglob("*.md"))
        + [root / "results" / "README.md"]
        + [root / "archive" / "README.md"]
    )
    missing = []
    for document in documents:
        link_parent = snapshot_origins.get(document, document).parent
        text = document.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
            path = target.split("#", 1)[0]
            if not path or re.match(r"^[a-z]+://", path) or path.startswith("mailto:"):
                continue
            if not (link_parent / path).exists():
                missing.append(f"{document.relative_to(root)} -> {target}")
    assert not missing, "missing local documentation links:\n" + "\n".join(missing)


def test_findings_archive_covers_numbered_research_ledger():
    archive = root / "archive/pregrant-history"
    index = (archive / "findings" / "ARCHIVE.md").read_text(encoding="utf-8")
    findings = sorted(
        (archive / "findings").glob("[0-9]*_*.md"),
        key=lambda path: int(path.name.split("_", 1)[0]),
    )
    assert [int(path.name.split("_", 1)[0]) for path in findings] == list(range(len(findings)))
    assert all(f"({path.name})" in index for path in findings)


@pytest.mark.evidence
def test_evaluation_table_matches_checked_results():
    original = historical_repository()
    document = (original / "docs" / "evaluation.md").read_text(encoding="utf-8")
    for path in sorted((original / "results").glob("*/open_slm.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        scores = result["scores_percent"]
        row = (
            f"| {path.parent.name} | {scores['hellaswag']:.2f} | {scores['arc_easy']:.2f} | "
            f"{scores['arc_challenge']:.2f} | {scores['piqa']:.2f} | "
            f"{scores['arithmark_3']:.2f} | {scores['intelligence_index']:.2f} | "
            f"{scores['arithmark_2']:.2f} |"
        )
        assert row in document
