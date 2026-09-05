"""Audit historical and planned Speck Paper 1 baseline evidence."""

import argparse
import json
import subprocess
from pathlib import Path

from speck.common import base_dir
from speck.paper_baseline import audit_baselines


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--cache-root", type=Path, default=Path(base_dir()))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def repository_revision():
    root = Path(__file__).resolve().parents[1]
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError("baseline audit requires a clean repository")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main(argv=None):
    args = arguments(argv)
    report = audit_baselines(
        args.matrix,
        args.cache_root,
        repository_revision(),
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
