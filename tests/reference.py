"""Original research inputs, isolated from the maintained implementation.

Record tests read a detached original checkout. Software imports still resolve to the
maintained package, so behavior tests exercise the new code with retained fixtures.
The temporary worktree is removed when the test process exits.
"""

import atexit
import functools
import subprocess
import tempfile
from pathlib import Path

from speck.provenance.archive import load_archive, restore_checkout


@functools.lru_cache(maxsize=1)
def historical_repository():
    root, _ = load_archive()
    parent = tempfile.TemporaryDirectory(prefix="speck-reference-")
    checkout = restore_checkout(Path(parent.name) / "repository", root)

    def cleanup():
        subprocess.run(
            ["git", "-C", str(root), "worktree", "remove", "--force", str(checkout)],
            check=False,
            capture_output=True,
        )
        parent.cleanup()

    atexit.register(cleanup)
    return checkout
