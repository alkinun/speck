"""Use frozen inputs with the maintained implementation; never execute archived tests/code."""

import atexit
import functools
import subprocess
import tempfile
from pathlib import Path

from speck.provenance.archive import load_archive, restore_checkout


@functools.lru_cache(maxsize=2)
def _checkout(legacy):
    root, manifest = load_archive()
    parent = tempfile.TemporaryDirectory(prefix="speck-fixtures-")
    revision = manifest["legacy_revision"] if legacy else manifest["revision"]
    checkout = restore_checkout(Path(parent.name) / "repository", root, revision=revision)

    def cleanup():
        subprocess.run(
            ["git", "-C", str(root), "worktree", "remove", "--force", str(checkout)],
            check=False,
            capture_output=True,
        )
        parent.cleanup()

    atexit.register(cleanup)
    return checkout


def historical_repository():
    return _checkout(True)


def preparation_repository():
    return _checkout(False)
