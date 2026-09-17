"""Historical workflows remain recoverable without occupying the active tree."""

import json
import subprocess

import pytest

from speck.provenance.archive import locate_original, restore_checkout, verify_archive
from speck.provenance.repository import repository_root


def test_snapshot_is_complete_and_contains_preparation_receipts():
    result = verify_archive()
    assert result["status"] == "verified"
    for path in (
        "research/status.json",
        "scripts/tokenizer_pilot_screen.py",
        "results/data/fineweb-edu-e1s-token-stock-20260914.json",
    ):
        assert locate_original(path)["revision"] == result["revision"]


def test_restore_preserves_original_bytes_without_changing_current_checkout(tmp_path):
    root = repository_root()
    before = (root / "README.md").read_bytes()
    destination = restore_checkout(tmp_path / "history")
    try:
        result = json.loads(
            (destination / "results/data/fineweb-edu-e1s-token-stock-20260914.json").read_text()
        )
        assert result["tokens"] == 2306703052
        assert (destination / "research/flagship/STUDY.md").is_file()
        assert (root / "README.md").read_bytes() == before
        with pytest.raises(FileExistsError):
            restore_checkout(destination)
    finally:
        subprocess.run(
            ["git", "-C", str(root), "worktree", "remove", str(destination)],
            check=True,
            capture_output=True,
        )


@pytest.mark.parametrize("path", ["../secrets", "/absolute", "missing-file.json"])
def test_locate_rejects_outside_and_missing_paths(path):
    with pytest.raises((ValueError, FileNotFoundError)):
        locate_original(path)
