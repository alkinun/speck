"""Report Python sources whose Git blobs are referenced by checked evidence."""

import hashlib
import subprocess
from pathlib import Path


def _git(root, *args, check=True):
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=check,
        capture_output=True,
    )


def _tracked_paths(root, revision):
    output = _git(root, "ls-tree", "-r", "-z", "--name-only", revision).stdout
    return tuple(
        Path(value.decode())
        for value in output.split(b"\0")
        if value and value.decode().endswith(".py")
    )


def _blob(root, revision, path):
    return _git(root, "show", f"{revision}:{path.as_posix()}").stdout


def _references(root, revision, digest, source):
    result = _git(root, "grep", "-l", "-F", "-e", digest, revision, "--", check=False)
    if result.returncode not in {0, 1}:
        raise subprocess.CalledProcessError(
            result.returncode,
            result.args,
            output=result.stdout,
            stderr=result.stderr,
        )
    prefix = f"{revision}:"
    references = []
    for line in result.stdout.decode().splitlines():
        path = line.removeprefix(prefix)
        if path != source.as_posix():
            references.append(path)
    return tuple(sorted(references))


def source_pin_inventory(root, revision="HEAD"):
    """Return tracked Python blobs whose SHA-256 appears elsewhere in a Git tree."""

    root = Path(root).resolve()
    inventory = []
    for path in _tracked_paths(root, revision):
        digest = hashlib.sha256(_blob(root, revision, path)).hexdigest()
        references = _references(root, revision, digest, path)
        if references:
            inventory.append(
                {
                    "path": path.as_posix(),
                    "sha256": digest,
                    "references": references,
                }
            )
    return tuple(inventory)


def changed_source_pins(root, base="HEAD", target=None):
    """Return changed Python blobs that were evidence-bound in the base tree."""

    root = Path(root).resolve()
    command = ["diff", "--name-only", "--no-renames", "-z", base]
    if target is not None:
        command.append(target)
    command.extend(("--", "*.py"))
    changed = {Path(value.decode()) for value in _git(root, *command).stdout.split(b"\0") if value}
    if not changed:
        return ()

    pinned = []
    base_paths = set(_tracked_paths(root, base))
    for path in sorted(changed & base_paths):
        digest = hashlib.sha256(_blob(root, base, path)).hexdigest()
        references = _references(root, base, digest, path)
        if references:
            pinned.append(
                {
                    "path": path.as_posix(),
                    "sha256": digest,
                    "references": references,
                }
            )
    return tuple(pinned)
