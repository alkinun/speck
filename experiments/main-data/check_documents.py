"""Resolve every relative link and heading anchor between the repository's Markdown documents.

Figures are not restated across documents: plan.json owns the numbers, and the only prose
tables that render them are checked by check_program_plan.py. This check makes sure that
consolidating or renaming a document cannot strand a reference. It reads documents only.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Directories whose Markdown is not ours to govern.
SKIP = (".venv", "wandb", "node_modules", ".git", ".pytest_cache")


def _slug(heading: str) -> str:
    """Reproduce GitHub's heading-anchor rule: drop punctuation, then one hyphen per space."""
    text = re.sub(r"[^\w\s-]", "", heading.strip().lower())
    return text.replace(" ", "-")


def _anchors(document: Path) -> set[str]:
    """Collect heading anchors, ignoring shell comments inside code fences."""
    found: set[str] = set()
    fenced = False
    for line in document.read_text().splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        match = re.match(r"#{1,6}\s+(.*)", line)
        if match:
            found.add(_slug(match.group(1)))
    return found


def _documents() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*.md")
        if not any(skip in str(path.relative_to(ROOT)) for skip in SKIP)
    )


def validate() -> dict:
    documents = _documents()
    anchors = {document.resolve(): _anchors(document) for document in documents}
    mismatches: list[str] = []
    checked = 0
    for document in documents:
        text = document.read_text()
        for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", text):
            target = match.group(1)
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            path, _, fragment = target.partition("#")
            line = text[: match.start()].count("\n") + 1
            checked += 1
            resolved = (document.parent / path).resolve() if path else document.resolve()
            where = f"{document.relative_to(ROOT)}:{line}"
            if path and not resolved.exists():
                mismatches.append(f"{where}: link target does not exist: {target}")
            elif fragment and resolved in anchors and fragment not in anchors[resolved]:
                mismatches.append(f"{where}: no such heading anchor: {target}")
    if mismatches:
        raise ValueError("stranded links:\n  " + "\n  ".join(mismatches))
    return {"documents_scanned": len(documents), "links_resolved": checked}


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(validate(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
