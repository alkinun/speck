"""Verify preserved research bytes and recover their original Git checkout."""

import hashlib
import json
import subprocess
from pathlib import Path

from speck.provenance.repository import repository_root


def load_archive(root=None):
    root = Path(root) if root is not None else repository_root()
    manifest = json.loads((root / "archive/manifest.json").read_text())
    if manifest.get("format") != "speck_research_archive" or manifest.get("format_version") != 1:
        raise ValueError("unsupported research archive")
    return root, manifest


def _relative(root, value):
    path = Path(value)
    resolved = (root / path).resolve()
    if path.is_absolute() or not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"archive path escapes its root: {value}")
    return resolved


def verify_archive(root=None):
    """Check every preserved file against both its manifest and original Git blob."""
    root, manifest = load_archive(root)
    revision = manifest["revision"]
    tree = subprocess.check_output(["git", "-C", str(root), "ls-tree", "-r", "-z", revision])
    blobs = {}
    for record in tree.split(b"\0"):
        if record:
            metadata, name = record.split(b"\t", 1)
            blobs[name.decode()] = metadata.split()[2].decode()
    entries = manifest["files"]
    if len({entry["original_path"] for entry in entries}) != len(entries):
        raise ValueError("duplicate original archive paths")
    if set(blobs) != {entry["original_path"] for entry in entries}:
        raise ValueError("archive inventory does not cover the preserved Git tree")
    expected_paths = {
        _relative(root, entry["archived_path"]) for entry in entries if entry.get("archived_path")
    }
    actual_paths = {
        path.resolve() for path in (root / "archive/pregrant-history").rglob("*") if path.is_file()
    }
    if actual_paths != expected_paths:
        raise ValueError("physical archive files differ from the inventory")
    requests = "".join(blobs[entry["original_path"]] + "\n" for entry in entries)
    output = subprocess.check_output(
        ["git", "-C", str(root), "cat-file", "--batch"], input=requests.encode()
    )
    offset = 0
    preserved = 0
    for entry in entries:
        end = output.index(b"\n", offset)
        header = output[offset:end].split()
        size = int(header[2])
        original = output[end + 1 : end + 1 + size]
        offset = end + size + 2
        if hashlib.sha256(original).hexdigest() != entry["sha256"]:
            raise ValueError(f"original Git identity mismatch: {entry['original_path']}")
        if entry.get("archived_path"):
            path = _relative(root, entry["archived_path"])
            if not path.is_file() or path.read_bytes() != original:
                raise ValueError(f"archived bytes changed: {entry['original_path']}")
            preserved += 1
    return {"revision": revision, "inventoried_files": len(entries), "archived_files": preserved}


def restore_checkout(destination, root=None):
    """Create an explicitly requested detached worktree at the preserved revision."""
    root, manifest = load_archive(root)
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"restore destination already exists: {destination}")
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "worktree",
            "add",
            "--detach",
            str(destination),
            manifest["revision"],
        ],
        check=True,
    )
    return destination


def locate_original(original_path, root=None):
    """Locate an original artifact, or the last Git tree containing a removed tool."""
    root, manifest = load_archive(root)
    _relative(root, original_path)
    for entry in manifest["files"]:
        if entry["original_path"] == original_path:
            return {**entry, "revision": manifest["revision"]}
    removed = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "log",
            "-1",
            "--diff-filter=D",
            "--format=%H",
            manifest["revision"],
            "--",
            original_path,
        ],
        text=True,
    ).strip()
    if not removed:
        raise FileNotFoundError(f"no archived path or removal history: {original_path}")
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", f"{removed}^"], text=True
    ).strip()
    value = subprocess.check_output(["git", "-C", str(root), "show", f"{revision}:{original_path}"])
    return {
        "original_path": original_path,
        "revision": revision,
        "sha256": hashlib.sha256(value).hexdigest(),
    }
