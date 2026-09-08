"""Provide shared file hashing and atomic JSON report writes."""

import hashlib
import json
import os
import tempfile
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
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
            handle.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
