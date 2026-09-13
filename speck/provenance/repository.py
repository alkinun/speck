"""Locate a checkout without depending on an artifact's directory depth."""

from pathlib import Path


def repository_root(path=None):
    """Find the nearest project root containing the Speck package and project metadata."""
    path = Path(path or __file__).resolve()
    start = path if path.is_dir() else path.parent
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "speck").is_dir():
            return candidate
    raise ValueError(f"cannot locate the Speck repository from {path}")


def repository_artifact(value, root=None):
    """Resolve a data/protocol input at its current or preserved original path.

    Callers retain their normal digest checks. Original Python source is recovered
    through Git, rather than silently substituted into current execution.
    """
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    root = Path(root) if root is not None else repository_root()
    current = root / path
    if current.exists():
        return current
    archived = root / "archive/pregrant-history" / path
    return archived if archived.exists() else current
