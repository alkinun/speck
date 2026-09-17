"""The maintained reading path must resolve without restoring historical plans."""

import re

from speck.provenance.repository import repository_root


def test_local_markdown_links_exist():
    root = repository_root()
    documents = [
        *root.glob("*.md"),
        *(root / "docs").rglob("*.md"),
        *(root / "experiments").rglob("*.md"),
        root / "archive/README.md",
    ]
    missing = []
    for document in documents:
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", document.read_text()):
            path = target.split("#", 1)[0]
            if not path or re.match(r"^[a-z]+://", path) or path.startswith("mailto:"):
                continue
            if not (document.parent / path).exists():
                missing.append(f"{document.relative_to(root)} -> {target}")
    assert not missing, "missing local documentation links:\n" + "\n".join(missing)
