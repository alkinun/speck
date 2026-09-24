"""Provide shared file hashing and atomic JSON report writes."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_reference(entry, root=REPOSITORY):
    """Resolve one record reference and verify it.

    Repository files are referenced by path alone: Git already versions them, and a copied
    digest or status goes stale on the next edit. Files outside the repository carry a digest.
    """
    path = Path(entry["path"])
    resolved = path if path.is_absolute() else root / path
    if resolved.resolve().is_relative_to(Path(root).resolve()):
        if {"sha256", "status", "format"} & entry.keys():
            raise ValueError(f"repository reference must be path-only: {path}")
        if not resolved.is_file():
            raise ValueError(f"missing referenced file: {path}")
    elif file_sha256(resolved) != entry.get("sha256"):
        raise ValueError(f"artifact checksum mismatch: {resolved}")
    return resolved


def lines_sha256(values):
    """Hash an ordered list with the dataset manifest's newline convention."""
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def fsync_path(path):
    """Flush a written file to disk, so a later rename or receipt cannot outlive its bytes."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value, *, fsync=False):
    """Replace a report only after serialization and writing have succeeded.

    Each writer owns a temporary file in the destination directory so concurrent
    reports cannot overwrite each other's staging files. JSON bytes retain the
    existing report convention: sorted keys, two-space indentation, and a newline.
    """

    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(payload)
            if fsync:
                handle.flush()
                os.fsync(handle.fileno())
            handle.close()
            os.replace(temporary, path)
            if fsync:
                descriptor = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        finally:
            temporary.unlink(missing_ok=True)


def durable_json(path, value):
    """Publish sorted report JSON with both payload and directory fsync."""
    atomic_json(path, value, fsync=True)
