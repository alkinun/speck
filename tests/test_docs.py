import json
import re
from pathlib import Path

root = Path(__file__).parents[1]


def test_local_markdown_links_exist():
    # Numbered findings and the archived Paper 1 program reference tooling that was removed from
    # the tree at commit 68d3187; their links are historical and are not checked here.
    documents = (
        sorted(root.glob("*.md"))
        + sorted((root / "docs").rglob("*.md"))
        + [root / "experiments" / "README.md"]
        + [root / "findings" / "README.md"]
        + [root / "papers" / "README.md"]
        + [root / "research" / "README.md"]
        + sorted((root / "research" / "flagship").rglob("*.md"))
        + [root / "research" / "paper-1" / "README.md"]
        + [root / "results" / "README.md"]
    )
    missing = []
    for document in documents:
        text = document.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
            path = target.split("#", 1)[0]
            if not path or re.match(r"^[a-z]+://", path) or path.startswith("mailto:"):
                continue
            if not (document.parent / path).exists():
                missing.append(f"{document.relative_to(root)} -> {target}")
    assert not missing, "missing local documentation links:\n" + "\n".join(missing)


def test_findings_archive_covers_numbered_research_ledger():
    index = (root / "findings" / "ARCHIVE.md").read_text(encoding="utf-8")
    findings = sorted(
        (root / "findings").glob("[0-9]*_*.md"),
        key=lambda path: int(path.name.split("_", 1)[0]),
    )
    assert [int(path.name.split("_", 1)[0]) for path in findings] == list(range(len(findings)))
    assert all(f"({path.name})" in index for path in findings)


def test_evaluation_table_matches_checked_results():
    document = (root / "docs" / "evaluation.md").read_text(encoding="utf-8")
    for path in sorted((root / "results").glob("*/open_slm.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        scores = result["scores_percent"]
        row = (
            f"| {path.parent.name} | {scores['hellaswag']:.2f} | {scores['arc_easy']:.2f} | "
            f"{scores['arc_challenge']:.2f} | {scores['piqa']:.2f} | "
            f"{scores['arithmark_3']:.2f} | {scores['intelligence_index']:.2f} | "
            f"{scores['arithmark_2']:.2f} |"
        )
        assert row in document
