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
