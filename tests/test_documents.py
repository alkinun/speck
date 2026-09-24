"""Links and heading anchors between documents must resolve."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_documents", ROOT / "experiments/main-data/check_documents.py"
)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def test_committed_documents_resolve():
    assert checker.validate()["links_resolved"] > 0


def _isolated_tree(tmp_path, monkeypatch, documents):
    """Point the checker at a small synthetic tree so document-side faults can be tested."""
    for name, text in documents.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    monkeypatch.setattr(checker, "_documents", lambda: sorted(tmp_path.rglob("*.md")))


def test_a_stranded_link_is_rejected(tmp_path, monkeypatch):
    """Consolidating a document must not leave a reference pointing at the old path."""
    _isolated_tree(tmp_path, monkeypatch, {"a.md": "# A\n\nSee [b](b.md).\n"})
    with pytest.raises(ValueError, match="link target does not exist"):
        checker.validate()


def test_a_stranded_anchor_is_rejected(tmp_path, monkeypatch):
    """Renaming a heading must not leave a reference pointing at the old anchor."""
    _isolated_tree(
        tmp_path,
        monkeypatch,
        {
            "a.md": "# A\n\nSee [b](b.md#gone).\n",
            "b.md": "# B\n\n## Still here\n",
        },
    )
    with pytest.raises(ValueError, match="no such heading anchor"):
        checker.validate()


def test_a_heading_anchor_matches_github_slugging(tmp_path, monkeypatch):
    """An em dash is dropped and its surrounding spaces each become a hyphen."""
    assert checker._slug("Cagliostro v3 review — 2026-09-20") == "cagliostro-v3-review--2026-09-20"
    _isolated_tree(
        tmp_path,
        monkeypatch,
        {
            "a.md": "# A\n\nSee [b](b.md#review--2026-09-20).\n",
            "b.md": "# B\n\n## Review — 2026-09-20\n",
        },
    )
    assert checker.validate()["links_resolved"] == 1


def test_a_shell_comment_in_a_code_fence_is_not_a_heading(tmp_path, monkeypatch):
    """`# Retain configs...` in a ```bash block is a comment, not an anchor."""
    _isolated_tree(
        tmp_path,
        monkeypatch,
        {"b.md": "# B\n\n```bash\n# Not a heading\nls\n```\n"},
    )
    assert checker._anchors(tmp_path / "b.md") == {"b"}
    _isolated_tree(
        tmp_path,
        monkeypatch,
        {
            "a.md": "# A\n\nSee [b](b.md#not-a-heading).\n",
            "b.md": "# B\n\n```bash\n# Not a heading\nls\n```\n",
        },
    )
    with pytest.raises(ValueError, match="no such heading anchor"):
        checker.validate()
