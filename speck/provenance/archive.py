"""Keep historical workflows recoverable in Git, outside the active source tree."""

import json
import subprocess
from pathlib import Path, PurePosixPath

from speck.provenance.repository import repository_root


def _git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def load_archive(root=None):
    root = Path(root) if root is not None else repository_root()
    manifest = json.loads((root / "archive/snapshot.json").read_text())
    if manifest.get("format") != "speck_git_archive" or manifest.get("format_version") != 1:
        raise ValueError("unsupported Git archive")
    for key in ("revision", "tree", "legacy_revision"):
        value = manifest[key]
        if (
            not isinstance(value, str)
            or len(value) != 40
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("invalid archive object identity")
    return root, manifest


def verify_archive(root=None):
    """Verify the anchored commit/tree and the legacy fixture revision without restoring files."""
    root, manifest = load_archive(root)
    actual = _git(root, "rev-parse", manifest["revision"] + "^{tree}")
    if actual != manifest["tree"]:
        raise ValueError("archive tree identity differs")
    _git(root, "cat-file", "-e", manifest["legacy_revision"] + "^{commit}")
    # Check every reachable archived blob/tree, not just the commit name.
    subprocess.run(
        ["git", "-C", str(root), "fsck", "--no-reflogs", "--no-dangling", manifest["revision"]],
        check=True,
        capture_output=True,
    )
    return {"revision": manifest["revision"], "tree": actual, "status": "verified"}


def restore_checkout(destination, root=None, *, revision=None):
    root, manifest = load_archive(root)
    revision = revision or manifest["revision"]
    if revision not in {manifest["revision"], manifest["legacy_revision"]}:
        raise ValueError("restore requires a recorded archive revision")
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    subprocess.run(
        ["git", "-C", str(root), "worktree", "add", "--detach", str(destination), revision],
        check=True,
        capture_output=True,
    )
    return destination


def locate_original(original_path, root=None):
    root, manifest = load_archive(root)
    path = PurePosixPath(original_path)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("historical paths must be repository-relative")
    for revision in (manifest["revision"], manifest["legacy_revision"]):
        result = subprocess.run(
            ["git", "-C", str(root), "cat-file", "-e", f"{revision}:{path}"], capture_output=True
        )
        if result.returncode == 0:
            return {"revision": revision, "path": str(path), "git_object": f"{revision}:{path}"}
    raise FileNotFoundError(original_path)
